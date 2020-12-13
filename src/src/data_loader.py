#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_loader.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/9
# @Desc  : None

import os
import torch
import random
import pandas as pd
from src.tool.shared_tool import load_graph, load_category, get_n_city_grid
from src.src.generate_train_data import GenerateTrainDataHelper

from src.args import args


class DataLoader(object):
    def __init__(self, DEVICE):
        self.DEVICE = DEVICE
        # base info
        graph_entity_relation_list = ["small-category_grid", "grid_small-category", "grid_grid"]
        if args.use_category_ontology_diagram:
            graph_entity_relation_list.extend(["small-category_big-category", "big-category_small-category"])
        self.graph_entity_relation_to_ID = dict({v: torch.tensor(k, device=DEVICE)
                                                 for k, v in enumerate(graph_entity_relation_list)})

        self.n_city_grid = [get_n_city_grid(city_id) for city_id in range(len(args.city_list))]
        self.n_kg_relation = len(graph_entity_relation_list)
        self.big_category_dict, self.small_category_dict, self.n_big_category, self.n_small_category = load_category()

        # load data
        self.test_grids, self.target_type_ids = self._generate_test_data()
        self.city_graphs = []
        self.train_kge = []
        self.train_ncf = []

        for city_id, _ in enumerate(args.city_list):
            # load graph
            self.city_graphs.append(load_graph(city_id).to(self.DEVICE))

            # load train data
            train_data_helper = GenerateTrainDataHelper(city_id, self.city_graphs[city_id],
                                                        self.graph_entity_relation_to_ID, self.DEVICE)
            self.train_kge.append(train_data_helper.kge_train)
            self.train_ncf.append(train_data_helper.ncf_train)

        self.train_batch_kge_index, self.train_batch_ncf_index = self._generate_train_batch_index()

    def _generate_train_batch_index(self):
        kge_batch_index = []
        ncf_batch_index = []

        for city_data in self.train_kge:
            single_city_keg_batch_index = list(range(0, len(city_data)))
            random.shuffle(single_city_keg_batch_index)
            kge_batch_index.append(single_city_keg_batch_index)

        for city_data in self.train_ncf:
            single_city_ncf_batch_index = list(range(0, len(city_data)))
            random.shuffle(single_city_ncf_batch_index)
            ncf_batch_index.append(single_city_ncf_batch_index)

        return kge_batch_index, ncf_batch_index

    def _generate_test_data(self):
        data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[args.target_city_id])
        test_grids = pd.read_csv(os.path.join(data_dir, 'test_data.csv'), header=None)
        test_grids = torch.tensor(test_grids.values, device=self.DEVICE)

        target_type_ids = torch.tensor([self.small_category_dict[v]
                                        for v in args.small_cate_for_fitting_list], device=self.DEVICE)

        return test_grids, target_type_ids

    # def shuffle_train_data_index(self):
    #     for city_id in range(len(args.city_list)):
    #         random.shuffle(self.train_batch_kge_index[city_id])
    #         random.shuffle(self.train_batch_ncf_index[city_id])

    def get_train_keg_batch_data(self, city_id, index):
        return self.train_kge[city_id][index]

    def get_train_ncf_batch_data(self, city_id,  index):
        return self.train_ncf[city_id][index]

    def get_test_data(self):
        return self.test_grids, self.target_type_ids
