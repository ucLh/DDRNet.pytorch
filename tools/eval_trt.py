# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.optim

import _init_paths
import datasets
from datasets.mappilary import Mappilary
from config import config
from config import update_config
from core.function import validate_trt
from utils.utils import create_logger
from trtseg import Segmentator


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/mappilary/ddrnet23_slim.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    test_dataset = Mappilary(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=config.TEST.NUM_SAMPLES,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    test_sampler = get_sampler(test_dataset)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    model = Segmentator(config.TRT.ENGINE_PATH, config.TRT.COLORS_PATH,
                        config.TEST.IMAGE_SIZE[0], config.TEST.IMAGE_SIZE[1])

    _, mean_IoU, IoU_array = validate_trt(config,
            testloader, model)

    print(mean_IoU, IoU_array)        


if __name__ == '__main__':
    main()
