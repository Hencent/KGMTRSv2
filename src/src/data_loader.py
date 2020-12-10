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
from src.tool.shared_tool import load_graph
from src.preprocess.generate_train_data import GenerateTrainDataHelper

from src.args import args


class DataLoader(object):
    def __init__(self, DEVICE):
        self.DEVICE = DEVICE
        # base info
        graph_entity_relation_list = ["small-category_grid", "grid_small-category", "grid_grid"]
        if args.use_category_ontology_diagram:
            graph_entity_relation_list.extend(["small-category_big-big-category", "big-category_small-category"])
        graph_entity_relation_to_ID = dict({v: torch.tensor(k, device=DEVICE)
                                            for k, v in enumerate(graph_entity_relation_list)})

        # load data
        self.test_data = self._load_test_data()
        self.city_graphs = []
        self.train_kge = []
        self.train_ncf = []

        for city_id, _ in enumerate(args.city_list):
            # load graph
            self.city_graphs.append(load_graph(city_id).to(self.DEVICE))

            # load train data
            train_data_helper = GenerateTrainDataHelper(city_id, self.city_graphs[city_id],
                                                        graph_entity_relation_to_ID, self.DEVICE)
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

    def _load_test_data(self):
        data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[args.target_city_id])
        test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'), header=None)
        test_data = torch.tensor(test_data.values, device=self.DEVICE)
        return test_data

    def get_train_keg_batch_data(self, index):
        return self.train_kge[index]

    def get_train_ncf_batch_data(self, index):
        return self.train_ncf[index]

    def get_test_data(self):
        return self.test_data
