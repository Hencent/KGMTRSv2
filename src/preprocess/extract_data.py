#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : extract_data.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

import os
import random
import numpy as np
import pandas as pd
import csv
from src.tool.shared_tool import load_category, ensure_dir_exist

from src.args import args


class _CityInfoExtractHelper(object):
    def __init__(self, city_id, big_category_dict, small_category_dict):
        self.city_id = city_id
        self.big_category_dict, self.small_category_dict = big_category_dict, small_category_dict

        # extract grid info
        self.n_longitude, self.n_latitude, self.n_grid, self.grid_coordinate_scope, \
            self.grid_relations = self._extract_grid_info()

        # load data
        self.dianping_data = self._load_dianping_data()
        # TODO load multi-type info

        # extract category-grid info
        self.small_category_grid_relation = self._extract_category_gird_interaction_info()

        # test_data
        if self.city_id == args.target_city_id:
            self.test_data = self._extract_test_data()

        # save preprocessed data
        self._save_preprocessed_data()

    def _extract_grid_info(self):
        area_longitude_boundary = np.arange(args.city_range[self.city_id][0],
                                            args.city_range[self.city_id][2],
                                            args.city_size[self.city_id])
        area_latitude_boundary = np.arange(args.city_range[self.city_id][3],
                                           args.city_range[self.city_id][1],
                                           args.city_size[self.city_id])

        n_longitude = len(area_longitude_boundary) - 1
        n_latitude = len(area_latitude_boundary) - 1
        n_grid = n_longitude * n_latitude

        grid_coordinate_scope = []
        grid_relations = []  # 双向关系，a->b 和 b->a 的相邻关系都有。

        for i in range(n_latitude):
            for j in range(n_longitude):
                grid_coordinate_scope.append(
                    [i * n_longitude + j, area_latitude_boundary[i], area_latitude_boundary[i + 1],
                     area_longitude_boundary[j], area_longitude_boundary[j + 1]])

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= i + dx < n_latitude and 0 <= j + dy < n_longitude and not (dx == 0 and dy == 0):
                            grid_relations.append([i * n_longitude + j, (i + dx) * n_longitude + j + dy])

        return n_longitude, n_latitude, n_grid, grid_coordinate_scope, grid_relations

    def _load_dianping_data(self):
        data_dir = os.path.join(args.raw_data_dir, args.city_list[self.city_id])
        dianping_data_path = os.path.join(data_dir, 'dianping.csv')
        dianping_data = pd.read_csv(dianping_data_path, usecols=args.city_dianping_index_order[self.city_id])

        # format data
        dianping_data = dianping_data[dianping_data['status'] == 0].drop(columns='status')  # 筛出正常营业的店铺
        dianping_data['review_count'].fillna(-1, inplace=True)  # 将 review_count 为空值用 - 1填充
        dianping_data['review_count'] = dianping_data['review_count'].astype('int64')

        # remap category name into category ID
        dianping_data['big_category'] = dianping_data['big_category'].map(
            lambda x: self.big_category_dict[x])
        dianping_data['small_category'] = dianping_data['small_category'].map(
            lambda x: self.small_category_dict[x])

        return dianping_data

    def _extract_category_gird_interaction_info(self):
        small_category_grid_relation = []  # small category  &  grid  relation

        intentionally_ignored_cate_id_list = [self.small_category_dict[cat]
                                              for cat in args.intentionally_ignored_cate_list]

        # handle single-type data
        for item in self.dianping_data.itertuples():
            # filter test data
            if item.small_category in intentionally_ignored_cate_id_list:
                continue

            # get grid id
            grid_id = -1
            for idx in range(self.n_grid):
                if self.grid_coordinate_scope[idx][1] <= item.latitude <= self.grid_coordinate_scope[idx][2] and \
                        self.grid_coordinate_scope[idx][3] <= item.longitude <= self.grid_coordinate_scope[idx][4]:
                    grid_id = idx
                    break
            if grid_id < 0:  # POI 所处位置不在该城市的选定区域中
                continue

            small_category_grid_relation.append([item.small_category, grid_id])

        # handle multi-type data
        # TODO 等苗子佳学姐处理好

        return small_category_grid_relation

    def _supplement_test_data(self, pos_grids):
        test_data = []
        all_grids_set = set(list(range(0, self.n_grid)))
        pos_grids_set = set(pos_grids)
        neg_grids = list(all_grids_set - pos_grids_set)
        for pos_grid in pos_grids:
            random.shuffle(neg_grids)
            test_data.append([pos_grid] + neg_grids[: args.n_neg_grid])
        return test_data

    def _extract_test_data(self):
        test_pos_grids = []  # grid id in which there exists a true POI with target type

        test_target_type_id_list = [self.small_category_dict[cat] for cat in args.test_target_type_list]

        if args.test_data_mode == 0:  # 0: 单类型数据中的指定类型作为 test data
            print("| | |--use test data mode 0: choose from single type data.")
            for item in self.dianping_data.itertuples():
                # filter test data
                if item.small_category not in test_target_type_id_list:
                    continue

                # get grid id
                grid_id = -1
                for idx in range(self.n_grid):
                    if self.grid_coordinate_scope[idx][1] <= item.latitude <= self.grid_coordinate_scope[idx][2] and \
                            self.grid_coordinate_scope[idx][3] <= item.longitude <= self.grid_coordinate_scope[idx][4]:
                        grid_id = idx
                        break
                if grid_id < 0:  # POI 所处位置不在该城市的选定区域中
                    continue

                test_pos_grids.append(grid_id)
        else:  # 1: 多类型数据中的指定店名部分 test data
            print("| | |--use test data mode 1: choose from multi-type data.")
            # TODO 等苗子佳学姐的数据
            pass

        test_data = self._supplement_test_data(test_pos_grids)

        return test_data

    def _save_preprocessed_data(self):
        data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[self.city_id]) + '/'
        ensure_dir_exist(data_dir)

        # category & grid relation
        with open(os.path.join(data_dir, 'category_grid_relation.csv'), 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(["small category ID", "grid ID"])
            f_csv.writerows(self.small_category_grid_relation)

        # geographical info
        with open(os.path.join(data_dir, 'grid_relations.csv'), 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(["grid source ID", "grid target ID"])
            f_csv.writerows(self.grid_relations)

        with open(os.path.join(data_dir, 'grid_coordinate_scope.csv'), 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(["grid ID", "lat min", "lat max", "lon min", "lon max"])
            f_csv.writerows(self.grid_coordinate_scope)

        # base info
        with open(os.path.join(data_dir, 'base_info.csv'), 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(["n_longitude", self.n_longitude])
            f_csv.writerow(["n_latitude", self.n_latitude])
            f_csv.writerow(["n_grid", self.n_grid])

        if self.city_id == args.target_city_id:
            with open(os.path.join(data_dir, 'test_data.csv'), 'w') as f:
                f_csv = csv.writer(f)
                f_csv.writerows(self.test_data)


def data_extract_and_generate_test_data():
    # load category
    big_category_dict, small_category_dict, n_big_category, n_small_category = load_category()

    # preprocess every city
    for city_id, city_name in enumerate(args.city_list):
        print("| |--  loading and preprocessing {} data.".format(city_name))
        _CityInfoExtractHelper(city_id, big_category_dict, small_category_dict)

