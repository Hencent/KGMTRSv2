#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : generate_train_data.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

import os
import dgl
import random
import pandas as pd

from src.args import args


class GenerateTrainDataHelper(object):
    # TODO 每一个 epoch 应该重新生成负样例
    def __init__(self, city_id, city_graph, graph_entity_relation_to_ID, DEVICE):
        self.city_id = city_id

        # load graph
        self.city_graph = city_graph

        # generate train data for KGE
        self.kge_train = self._generate_train_data_for_KGE(graph_entity_relation_to_ID, DEVICE)

        # generate train data for NCF
        self.ncf_train = self._generate_train_data_for_NCF()

    def _generate_train_data_for_KGE(self, graph_entity_relation_to_ID, DEVICE):
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
        batch_data_for_kg_transR = []
        for edge_type in self.city_graph.etypes:
            head, pos_tail, eid = self.city_graph.edges(etype=edge_type, form='all')
            neg_tail = list(neg_sampler(self.city_graph, {edge_type: eid}).values())[0][1]

            # 打乱顺序，防止同类型边每次都是相邻格子
            index_for_shuffle = list(range(len(head)))
            random.shuffle(index_for_shuffle)
            head, pos_tail, neg_tail = head[index_for_shuffle], pos_tail[index_for_shuffle], neg_tail[index_for_shuffle]

            head, pos_tail, neg_tail = head.to(DEVICE), pos_tail.to(DEVICE), neg_tail.to(DEVICE)
            batch_data_for_kg_transR.extend([[graph_entity_relation_to_ID[edge_type],
                                              head[i:i + args.kg_transR_batch_size],
                                              pos_tail[i:i + args.kg_transR_batch_size],
                                              neg_tail[i:i + args.kg_transR_batch_size]]
                                             for i in range(0, len(head), args.kg_transR_batch_size)])

        return batch_data_for_kg_transR

    def _generate_train_data_for_NCF(self):
        cate_list = []
        pos_grid_list = []
        neg_grid_list = []

        # single category
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
        category, pos_grid, eid = self.city_graph.edges(etype="category_grid", form='all')
        neg_grid = list(neg_sampler(self.city_graph, {"category_grid": eid}).values())[0][1]

        category = category.tolist()
        pos_grid = pos_grid.tolist()
        neg_grid = neg_grid.tolist()

        cate_list.extend([[cat] for cat in category])
        pos_grid_list.extend(pos_grid)
        neg_grid_list.extend(neg_grid)

        # multi-category
        # TODO 移到 extract data 里面完成？
        pass

        batch_data_for_NCF = [[cate_list[i: i + args.NCF_batch_size],
                               pos_grid_list[i: i + args.NCF_batch_size],
                               neg_grid_list[i: i + args.NCF_batch_size]]
                              for i in range(0, len(cate_list), args.NCF_batch_size)]

        return batch_data_for_NCF
