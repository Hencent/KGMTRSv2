#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : system_init.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/10
# @Desc  : None


import random
import numpy as np
import torch
from src.src.log_tool import log_tool_init

from src.args import args


def system_init():
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # init log
    log_tool_init(folder=args.save_dir, no_console=False)
