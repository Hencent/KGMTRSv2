#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

import torch
from src.src.data_loader import DataLoader
from src.src.system_init import system_init
from src.src.log_tool import logging

from src.args import args

CUDA_AVAILABLE = False
DEVICE = None
N_GPU = 0


if __name__ == '__main__':
    # system init and CUDA
    if args.use_category_ontology_diagram:
        print("Using category ontology diagram")
    else:
        print("Not using category ontology diagram")

    system_init()

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(args.seed)
        N_GPU = torch.cuda.device_count()
    DEVICE = torch.device("cuda:{}".format(0) if CUDA_AVAILABLE else "cpu")
    logging.info("|--           parse args and init done.")

    data = DataLoader(DEVICE)
    logging.info("|--           load data.")


