# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import logging
import pathlib
from typing import List, Callable

import warnings

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.callbacks import BaseCallback, PerformanceCallback
from se3_transformer.runtime.loggers import LoggerCollection, DLLogger, WandbLogger, Logger
from se3_transformer.runtime.training import print_parameters_count
from se3_transformer.runtime.utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity

from trip.data_loading import GraphConstructor, TrIPDataModule
from trip.model import TrIP
from trip.runtime.arguments import PARSER
from trip.runtime.callbacks import TrIPMetricCallback, TrIPLRSchedulerCallback
from trip.runtime.inference import evaluate


warnings.filterwarnings("ignore", message=r"Non-finite norm encountered", category=FutureWarning)

def save_state(model: nn.Module, epoch: int, path: pathlib.Path, callbacks: List[BaseCallback]):
    """ Saves model, optimizer and epoch states to path (only once per node) """
    if get_local_rank() == 0:
        module = model.module if isinstance(model, DistributedDataParallel) else model
        checkpoint = module.get_checkpoint(epoch)

        for callback in callbacks:
            callback.on_checkpoint_save(checkpoint)

        torch.save(checkpoint, str(path))
        logging.info(f'Saved checkpoint to {str(path)}')


def load_state(model: nn.Module, path: pathlib.Path, callbacks: List[BaseCallback]):
    map_location = {'cuda:0': f'cuda:{get_local_rank()}'}
    trip_model = model.module if isinstance(model, DistributedDataParallel) else model
    checkpoint = trip_model.load_state(path, map_location)

    for callback in callbacks:
        callback.on_checkpoint_load(checkpoint)

    logging.info(f'Loaded checkpoint from {str(path)}')
    return checkpoint['epoch']


def train_epoch(model, graph_constructor, add_atom_data, train_dataloader, error_fn, loss_fn,
                epoch_idx, grad_scaler, optimizer, local_rank, callbacks, args):
    # TODO: Find a cleaner way to deal with accumulation variables
    energy_loss_acc = torch.zeros((1,), device='cuda')
    forces_loss_acc = torch.zeros((1,), device='cuda')
    energy_error_acc = torch.zeros((1,), device='cuda')
    forces_error_acc = torch.zeros((1,), device='cuda')
    num_atoms_acc = torch.zeros((1,), device='cuda')
    num_confs_acc = torch.zeros((1,), device='cuda')
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit='batch',
                         desc=f'Epoch {epoch_idx}', disable=(args.silent or local_rank != 0)):
        num_atoms = 0
        if args.add_atoms:
            batch, num_atoms = add_atom_data(*batch)
        species, pos_list, energy, forces, boxsize = to_cuda(batch)
        target = energy, forces
        graph = graph_constructor.create_graphs(pos_list, boxsize)
        graph.ndata['species'] = species

        for callback in callbacks:
            callback.on_batch_start()

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(graph, create_graph=True, standardized=True)
            energy_loss, forces_loss = loss_fn(pred, target)
            energy_error, forces_error = error_fn(pred, target, num_atoms)
            energy_loss /= args.accumulate_grad_batches
            forces_loss /= args.accumulate_grad_batches
            loss = energy_loss + args.force_weight*forces_loss
            
        energy_loss_acc += energy_loss.detach()
        forces_loss_acc += forces_loss.detach()
        energy_error_acc += energy_error.detach()
        forces_error_acc += forces_error.detach()
        num_atoms_acc += len(forces) - num_atoms
        num_confs_acc += len(energy) - num_atoms
        grad_scaler.scale(loss).backward()

        # gradient accumulation
        if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(train_dataloader):
            if args.gradient_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            model.zero_grad(set_to_none=True)

    energy_loss = energy_loss_acc / (i + 1)
    forces_loss = forces_loss_acc / (i + 1)
    energy_error = energy_error_acc / num_confs_acc
    forces_error = forces_error_acc / num_atoms_acc / 3

    return energy_loss, forces_loss, energy_error, forces_error


def log_epoch(energy_loss, forces_loss, energy_error, forces_error, logger, device,
              energy_std, world_size, epoch_idx):
    if dist.is_initialized():
        torch.distributed.all_reduce(energy_loss)
        torch.distributed.all_reduce(forces_loss)
        torch.distributed.all_reduce(energy_error)
        torch.distributed.all_reduce(forces_error)
        energy_loss /= world_size
        forces_loss /= world_size
        energy_error /= world_size
        forces_error /= world_size

    energy_loss = energy_loss.item()
    forces_loss = forces_loss.item()
    energy_error = energy_error.item()
    forces_error = forces_error.item()

    factor = energy_std * 627.5
    energy_error = np.sqrt(energy_error) * factor
    forces_error = np.sqrt(forces_error) * factor

    logging.info(f'Energy loss: {energy_loss:.5f}')
    logging.info(f'Forces loss: {forces_loss:.5f}')
    logging.info(f'Energy error: {energy_error:.3f}')
    logging.info(f'Forces error: {forces_error:.3f}')
    logger.log_metrics({'energy loss': energy_loss}, epoch_idx)
    logger.log_metrics({'forces loss': forces_loss}, epoch_idx)
    logger.log_metrics({'energy error': energy_error}, epoch_idx)
    logger.log_metrics({'forces error': forces_error}, epoch_idx)


def train(model: nn.Module,
          optimizer: Optimizer,
          graph_constructor: GraphConstructor,
          add_atom_data: Callable,
          error_fn: Callable,
          loss_fn: _Loss,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          energy_std: float,
          callbacks: List[BaseCallback],
          logger: Logger,
          args):
    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model._set_static_graph()

    model.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    epoch_start = load_state(model, args.load_ckpt_path, callbacks) if args.load_ckpt_path else 0

    for callback in callbacks:
        callback.on_fit_start(optimizer, args)

    for epoch_idx in range(epoch_start, args.epochs):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        train_output = train_epoch(
                model, graph_constructor, add_atom_data, train_dataloader, error_fn,
                loss_fn, epoch_idx, grad_scaler, optimizer, local_rank, callbacks, args)

        log_epoch(*train_output, logger, device, energy_std, world_size, epoch_idx)

        for callback in callbacks:
            callback.on_epoch_end()

        if not args.benchmark and args.save_ckpt_path is not None and args.ckpt_interval > 0 \
                and (epoch_idx + 1) % args.ckpt_interval == 0:
            save_state(model, epoch_idx, args.save_ckpt_path, callbacks)

        if not args.benchmark and (
                (args.eval_interval > 0 and (epoch_idx + 1) % args.eval_interval == 0) or epoch_idx + 1 == args.epochs):
            evaluate(model, graph_constructor, val_dataloader, callbacks, args)
            model.train()

            for callback in callbacks:
                callback.on_validation_end(epoch_idx)

    if args.save_ckpt_path is not None and not args.benchmark:
        save_state(model, args.epochs, args.save_ckpt_path, callbacks)

    for callback in callbacks:
        callback.on_fit_end()




if __name__ == '__main__':
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args = PARSER.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO)

    logging.info('============ TrIP =============')
    logging.info('|      Training procedure     |')
    logging.info('===============================')

    if args.seed is not None:
        logging.info(f'Using seed {args.seed}')
        seed_everything(args.seed)

    loggers = [DLLogger(save_dir=args.log_dir, filename=args.dllogger_name)]
    if args.wandb:
        loggers.append(WandbLogger(name=f'TrIP', save_dir=args.log_dir, project='trip'))
    logger = LoggerCollection(loggers)

    datamodule = TrIPDataModule(**vars(args))
    energy_std = datamodule.energy_std.item()
    logging.info(f'Dataset energy std: {energy_std:.5f}')

    model = TrIP(
        energy_std=energy_std,
        si_tensor=datamodule.si_tensor,
        tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively,
        **vars(args)
    )
    optimizer = TrIP.make_optimizer(model, **vars(args))
    graph_constructor = GraphConstructor(args.cutoff)
    add_atom_data = datamodule.add_atom_data
    error_fn = TrIP.error_fn
    loss_fn = TrIP.loss_fn

    if args.benchmark:
        logging.info('Running benchmark mode')
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        callbacks = [PerformanceCallback(logger, args.batch_size * world_size)]
    else:
        callbacks = [TrIPMetricCallback(logger, targets_std=energy_std, prefix='energy validation'),
                     TrIPMetricCallback(logger, targets_std=energy_std, prefix='forces validation'),
                     TrIPLRSchedulerCallback(logger)]

    if is_distributed:
        gpu_affinity.set_affinity(gpu_id=get_local_rank(), nproc_per_node=torch.cuda.device_count())

    print_parameters_count(model)
    logger.log_hyperparams(vars(args))
    increase_l2_fetch_granularity()
    train(model,
          optimizer,
          graph_constructor,
          add_atom_data,
          error_fn,
          loss_fn,
          datamodule.train_dataloader(),
          datamodule.val_dataloader(),
          energy_std,
          callbacks,
          logger,
          args)

    logging.info('Training finished successfully')
