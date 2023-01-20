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

from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.callbacks import BaseCallback
from se3_transformer.runtime.loggers import DLLogger, WandbLogger, LoggerCollection
from se3_transformer.runtime.utils import to_cuda, get_local_rank

from trip.data_loading import GraphConstructor
from trip.runtime.arguments import PARSER
from trip.runtime.callbacks import TrIPMetricCallback
from trip.model import TrIP


def evaluate(model: nn.Module,
             graph_constructor: GraphConstructor,
             dataloader: DataLoader,
             callbacks: List[BaseCallback],
             args):
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit='batch', desc=f'Evaluation',
                         leave=False, disable=(args.silent or get_local_rank() != 0)):
        species, pos_list, energy, forces, box_size = to_cuda(batch)
        target = energy, forces
        graph = graph_constructor.create_graphs(pos_list, box_size)
        graph.ndata['species'] = species

        for callback in callbacks:
            callback.on_batch_start()

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(graph, create_graph=False, standardized=True)

            for callback in callbacks:
                callback.on_validation_step(graph, target, pred)


if __name__ == '__main__':
    from se3_transformer.runtime.callbacks import PerformanceCallback
    from se3_transformer.runtime.utils import init_distributed, seed_everything
    from trip.model import TrIP
    from trip.data_loading import TrIPDataModule
    import torch.distributed as dist
    import logging
    import sys

    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args = PARSER.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO)

    logging.info('============ TrIP =============')
    logging.info('|  Inference on the test set  |')
    logging.info('===============================')

    if not args.benchmark and args.load_ckpt_path is None:
        logging.error('No load_ckpt_path provided, you need to provide a saved model to evaluate')
        sys.exit(1)

    if args.benchmark:
        logging.info('Running benchmark mode with one warmup pass')

    if args.seed is not None:
        seed_everything(args.seed)

    major_cc, minor_cc = torch.cuda.get_device_capability()

    loggers = [DLLogger(save_dir=args.log_dir, filename=args.dllogger_name)]
    if args.wandb:
        loggers.append(WandbLogger(name=f'TrIP', save_dir=args.log_dir, project='trip'))
    logger = LoggerCollection(loggers)
    datamodule = TrIPDataModule(**vars(args))
    energy_std = datamodule.energy_std.item()

    graph_constructor = GraphConstructor(args.cutoff)
    model = TrIP.load(path=str(args.load_ckpt_path), 
                        map_location={'cuda:0': f'cuda:{local_rank}'})
    callbacks = [TrIPMetricCallback(logger, targets_std=energy_std, prefix='energy'),
                 TrIPMetricCallback(logger, targets_std=energy_std, prefix='forces')]

    if is_distributed:
        nproc_per_node = torch.cuda.device_count()
        affinity = gpu_affinity.set_affinity(local_rank, nproc_per_node)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model._set_static_graph()

    test_dataloader = datamodule.test_dataloader() if not args.benchmark else datamodule.train_dataloader()
    evaluate(model,
             graph_constructor,
             test_dataloader,
             callbacks,
             args)

    for callback in callbacks:
        callback.on_validation_end()

    if args.benchmark:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        callbacks = [PerformanceCallback(logger, args.batch_size * world_size, warmup_epochs=1, mode='inference')]
        for _ in range(6):
            evaluate(model,
                     graph_constructor,
                     test_dataloader,
                     callbacks,
                     args)
            callbacks[0].on_epoch_end()

        callbacks[0].on_fit_end()
