#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : metrics.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/13
# @Desc  : None

import math


def hit_ratio_at_K(v, predict_list, K):
    if v in predict_list[:K]:
        return 1
    else:
        return 0


def ndcg_at_K(v, predict_list, K):
    for i in range(K):
        if v == predict_list[i]:
            return 1 / math.log2(i+2)
    else:
        return 0
