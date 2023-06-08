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
import warnings

import torch
import torch.distributed as dist

from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.callbacks import PerformanceCallback
from se3_transformer.runtime.loggers import LoggerCollection, DLLogger, WandbLogger
from se3_transformer.runtime.training import print_parameters_count
from se3_transformer.runtime.utils import get_local_rank, init_distributed, seed_everything, \
    increase_l2_fetch_granularity

from trip.data_loading import GraphConstructor, TrIPDataModule
from trip.model import TrIP
from trip.runtime.arguments import PARSER
from trip.runtime.callbacks import TrIPMetricCallback, TrIPLRSchedulerCallback
from trip.runtime.training import train

warnings.filterwarnings("ignore", message=r"Non-finite norm encountered", category=FutureWarning)



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

    model = TrIP.load(args.load_ckpt_path)
    args.load_ckpt_path = None  # Don't want to use previous 
    model.energy_std = energy_std
    model.si_tensor = datamodule.si_tensor
    model.cutoff = args.cutoff

    optimizer = TrIP.make_optimizer(model, **vars(args))
    graph_constructor = GraphConstructor(args.cutoff)
    #add_atom_data = datamodule.add_atom_data
    def add_atom_data(*vars): # For now, don't include atoms
        return tuple(vars), 0
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
