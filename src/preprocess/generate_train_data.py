#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : generate_train_data.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

import os
import dgl
import random

from src.args import args


class GenerateTrainDataHelper(object):
    def __init__(self, city_id, city_graph, graph_entity_relation_to_ID, DEVICE):
        self.city_id = city_id
        # load graph
        # self.city_graph = load_graph(city_id)
        self.city_graph = city_graph

        # generate train data for KGE
        self.kge_train = self._generate_train_data_for_KGE(graph_entity_relation_to_ID, DEVICE)

        # generate train data for NCF
        self.ncf_train = self._generate_train_data_for_NCF(DEVICE)

    def _generate_train_data_for_KGE(self, graph_entity_relation_to_ID, DEVICE):
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
        batch_data_for_kg_transR = []
        for edge_type in self.city_graph.etypes:
            head, pos_tail, eid = self.city_graph.edges(etype=edge_type, form='all')
            neg_tail = list(neg_sampler(self.city_graph, {edge_type: eid}).values())[0][1]

            # 打乱顺序，防止每次都是相邻格子
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

    def _generate_train_data_for_NCF(self, DEVICE):
        cate_list = []
        pos_grid_list = []
        neg_grid_list = []

        # single category
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
        category, pos_grid, eid = self.city_graph.edges(etype="small-category_grid", form='all')
        neg_grid = list(neg_sampler(self.city_graph, {"small-category_grid": eid}).values())[0][1]

        cate_list.extend([[cat] for cat in category])
        pos_grid_list.extend(pos_grid)
        neg_grid_list.extend(neg_grid)

        # multi-category
        data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[self.city_id])
        # TODO 等待苗子佳学姐的数据
        pass

        batch_data_for_kg_transR = [[cate_list[i: i + args.NCF_batch_size],
                                     pos_grid_list[i: i + args.NCF_batch_size],
                                     neg_grid_list[i: i + args.NCF_batch_size]]
                                    for i in range(0, len(cate_list), args.NCF_batch_size)]

        return batch_data_for_kg_transR
