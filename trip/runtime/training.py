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
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from apex.optimizers import FusedAdam, FusedLAMB
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from trip.model import TrIP
from se3_transformer.model.fiber import Fiber
from se3_transformer.runtime import gpu_affinity
from trip.runtime.arguments import PARSER
from trip.runtime.callbacks import TrIPMetricCallback, TrIPLRSchedulerCallback
from se3_transformer.runtime.callbacks import BaseCallback, PerformanceCallback
from trip.runtime.inference import evaluate
from se3_transformer.runtime.loggers import LoggerCollection, DLLogger, WandbLogger, Logger
from se3_transformer.runtime.utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity
from trip.data_loading import GraphConstructor, TrIPDataModule

from se3_transformer.runtime.training import save_state, load_state, print_parameters_count


def train_epoch(model, graph_constructor, train_dataloader, loss_fn, epoch_idx,
                grad_scaler, optimizer, local_rank, callbacks, args):
    energy_losses = []
    forces_losses = []
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit='batch',
                         desc=f'Epoch {epoch_idx}', disable=(args.silent or local_rank != 0)):
        species, pos_list, box_size_list, target = to_cuda(batch)
        graph = graph_constructor.create_graphs(pos_list, box_size_list)
        graph.edata['species'] = species

        for callback in callbacks:
            callback.on_batch_start()

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(graph, create_graph=True)
            energy_loss, forces_loss = loss_fn(pred, target)
            energy_loss /= args.accumulate_grad_batches
            forces_loss /= args.accumulate_grad_batches
            loss = energy_loss + args.force_weight*forces_loss
        grad_scaler.scale(loss).backward()

        # gradient accumulation
        if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(train_dataloader):
            if args.gradient_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            model.zero_grad(set_to_none=True)

        energy_losses.append(energy_loss.item())
        forces_losses.append(forces_loss.item())

    return np.mean(energy_losses), np.mean(forces_losses)


def train(model: nn.Module,
          graph_constructor: GraphConstructor,
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
    if args.optimizer == 'adam':
        optimizer = FusedAdam(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),
                              weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    epoch_start = load_state(model, optimizer, args.load_ckpt_path, callbacks) if args.load_ckpt_path else 0

    for callback in callbacks:
        callback.on_fit_start(optimizer, args)

    for epoch_idx in range(epoch_start, args.epochs):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        energy_loss, forces_loss = train_epoch(model, graph_constructor, train_dataloader, loss_fn, epoch_idx,
                                               grad_scaler, optimizer, local_rank, callbacks, args)
        if dist.is_initialized():
            energy_loss = torch.tensor(energy_loss, dtype=torch.float, device=device)
            torch.distributed.all_reduce(energy_loss)
            energy_loss = (energy_loss / world_size).item()
            forces_loss = torch.tensor(forces_loss, dtype=torch.float, device=device)
            torch.distributed.all_reduce(forces_loss)
            forces_loss = (forces_loss / world_size).item()

        factor = energy_std * 627.5
        energy_error = np.sqrt(energy_loss) * factor
        forces_error = np.sqrt(forces_loss) * factor

        logging.info(f'Energy error: {energy_error:.3f}')
        logging.info(f'Forces error: {forces_error:.3f}')
        logger.log_metrics({'energy error': energy_error}, epoch_idx)
        logger.log_metrics({'forces error': forces_error}, epoch_idx)

        for callback in callbacks:
            callback.on_epoch_end()

        if not args.benchmark and args.save_ckpt_path is not None and args.ckpt_interval > 0 \
                and (epoch_idx + 1) % args.ckpt_interval == 0:
            save_state(model, optimizer, epoch_idx, args.save_ckpt_path, callbacks)

        if not args.benchmark and (
                (args.eval_interval > 0 and (epoch_idx + 1) % args.eval_interval == 0) or epoch_idx + 1 == args.epochs):
            evaluate(model, val_dataloader, callbacks, args)
            model.train()

            for callback in callbacks:
                callback.on_validation_end(epoch_idx)

    if args.save_ckpt_path is not None and not args.benchmark:
        save_state(model, optimizer, args.epochs, args.save_ckpt_path, callbacks)

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
    energy_std = datamodule.get_energy_std()

    graph_constructor = GraphConstructor(args.cutoff)
    model = TrIP(
        fiber_in=Fiber({0: datamodule.NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: args.num_basis_fns}),
        output_dim=1,
        tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively,
        **vars(args)
    )
    loss_fn = datamodule.loss_fn

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
          loss_fn,
          datamodule.train_dataloader(),
          datamodule.val_dataloader(),
          energy_std,
          callbacks,
          logger,
          args)

    logging.info('Training finished successfully')
