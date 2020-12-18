#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : shared_tool.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

import os
import csv
import dgl
import pandas as pd

from src.args import args


def load_category():
    big_category = pd.read_csv(os.path.join(args.category_dir, 'big_category.csv'))
    small_category = pd.read_csv(os.path.join(args.category_dir, 'small_category.csv'))

    big_category_dict = dict()
    small_category_dict = dict()
    for item in big_category.itertuples():
        big_category_dict[item.name] = item.ID
    for item in small_category.itertuples():
        small_category_dict[item.name] = item.ID

    bcID2globalID = pd.read_csv(os.path.join(args.category_dir, 'big_cat_ID_to_global_ID.csv'))
    scID2globalID = pd.read_csv(os.path.join(args.category_dir, 'small_cat_ID_to_global_ID.csv'))
    big_cate_ID_to_global_ID_dict = dict()
    small_cate_ID_to_global_ID_dict = dict()
    for item in bcID2globalID.itertuples():
        big_cate_ID_to_global_ID_dict[item.big_cate_id] = item.global_cate_id
    for item in scID2globalID.itertuples():
        small_cate_ID_to_global_ID_dict[item.small_cate_id] = item.global_cate_id

    multi_level_cate = pd.read_csv(os.path.join(args.category_dir, 'cate_with_multi_level.csv')).values.tolist()
    multi_level_cate = [item[0] for item in multi_level_cate]

    n_category = max(bcID2globalID['global_cate_id'].max(), scID2globalID['global_cate_id'].max())

    return big_category_dict, small_category_dict, big_cate_ID_to_global_ID_dict, \
        small_cate_ID_to_global_ID_dict, multi_level_cate, n_category


def ensure_dir_exist(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


# def load_category_relation():
#     data_dir = os.path.join(args.category_dir, "big_small_category_relation.csv")
#     relation_data = pd.read_csv(data_dir)
#     return relation_data


def load_category_relation():
    data_dir = os.path.join(args.category_dir, "cate_relation.csv")
    relation_data = pd.read_csv(data_dir)
    return relation_data


def load_graph(city_id):
    data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[city_id])
    city_graph, _ = dgl.load_graphs(os.path.join(data_dir, "city_graph"))
    return city_graph[0]


def load_city_base_info(city_id):
    data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[city_id])
    with open(os.path.join(data_dir, "base_info.csv"), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        n_longitude, n_latitude, n_grid = [int(row[1]) for row in reader]
    return n_longitude, n_latitude, n_grid


def get_n_city_grid(city_id):
    _, _, n_grid = load_city_base_info(city_id)
    return n_grid

