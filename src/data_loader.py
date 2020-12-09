#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_loader.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/9
# @Desc  : None

import torch
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
        # self.test_data = self._load_test_data()
        self.city_graphs = []
        self.train_kge = []
        self.train_ncf = []

        for city_id, _ in enumerate(args.city_list):
            # load graph
            self.city_graphs.append(load_graph(city_id))

            # load train data
            train_data_helper = GenerateTrainDataHelper(city_id, self.city_graphs[city_id], graph_entity_relation_to_ID)
            self.train_kge.append(train_data_helper.kge_train)
            self.train_ncf.append(train_data_helper.ncf_train)

    # def _load_test_data(self):
