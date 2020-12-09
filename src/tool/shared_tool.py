#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : shared_tool.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

import os
import dgl
import pandas as pd

from src.args import args


def load_category():
    big_category_dir = os.path.join(args.category_dir, 'big_category.csv')
    small_category_dir = os.path.join(args.category_dir, 'small_category.csv')
    big_category = pd.read_csv(big_category_dir)
    small_category = pd.read_csv(small_category_dir)

    big_category_dict = dict()
    small_category_dict = dict()
    for item in big_category.itertuples():
        big_category_dict[item.name] = item.ID
    for item in small_category.itertuples():
        small_category_dict[item.name] = item.ID

    return big_category_dict, small_category_dict, len(big_category_dict), len(small_category_dict)


def ensure_dir_exist(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def load_category_relation():
    data_dir = os.path.join(args.category_dir, "big_small_category_relation.csv")
    relation_data = pd.read_csv(data_dir)
    return relation_data


def load_graph(city_id):
    data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[city_id])
    city_graph, _ = dgl.load_graphs(os.path.join(data_dir, "city_graph"))
    return city_graph[0]
