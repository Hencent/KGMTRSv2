#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : KGMTRS.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/10
# @Desc  : None
from typing import Any

import torch
import torch.nn as nn
import dgl

from src.args import args


class KnowledgeGraph(nn.Module):

    def __init__(self, n_big_category, n_small_category, n_city_grid, n_relation):
        super().__init__()
        self.small_category_embedding = nn.Parameter(torch.randn(n_small_category, args.embedding_size),
                                                     requires_grad=True)
        self.big_category_embedding = nn.Parameter(torch.randn(n_big_category, args.embedding_size),
                                                   requires_grad=True)
        self.graph_relation_embed = nn.Parameter(torch.randn(n_relation, args.relation_dim), requires_grad=True)
        self.graph_W_R = nn.Parameter(torch.rand([n_relation, args.embedding_size, args.relation_dim]),
                                      requires_grad=True)

        nn.init.xavier_uniform_(self.small_category_embedding, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.big_category_embedding, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.graph_relation_embed, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.graph_W_R, gain=nn.init.calculate_gain('leaky_relu'))

        self.city_grid_embedding = [nn.Parameter(torch.randn(n_grid, args.embedding_size), requires_grad=True)
                                    for n_grid in n_city_grid]
        self.city_bias = [nn.Parameter(torch.rand(args.city_bias_size), requires_grad=True) for _ in args.city_list]
        for city_id, _ in enumerate(args.city_list):
            self.register_parameter('city_grid_embedding_{}'.format(city_id), self.city_grid_embedding[city_id])
            self.register_parameter('city_bias_{}'.format(city_id), self.city_bias[city_id])
            nn.init.xavier_uniform_(self.city_grid_embedding[city_id], gain=nn.init.calculate_gain('leaky_relu'))
            nn.init.normal_(self.city_bias[city_id])

    def _get_graph_embedding_for_kg_transR(self, city_id, h, t_pos, t_neg, relation):
        h_embed = t_pos_embed = t_neg_embed = None

        if relation == 0:  # "small-category_grid"
            h_embed = self.small_category_embedding[h]
            t_pos_embed = self.city_grid_embedding[city_id][t_pos]
            t_neg_embed = self.city_grid_embedding[city_id][t_neg]
        elif relation == 1:  # "grid_small-category"
            h_embed = self.city_grid_embedding[city_id][h]
            t_pos_embed = self.small_category_embedding[t_pos]
            t_neg_embed = self.small_category_embedding[t_neg]
        elif relation == 2:  # "grid_grid":
            h_embed = self.city_grid_embedding[city_id][h]
            t_pos_embed = self.city_grid_embedding[city_id][t_pos]
            t_neg_embed = self.city_grid_embedding[city_id][t_neg]
        elif relation == 3:  # "small-category_big-big-category":
            h_embed = self.small_category_embedding[h]
            t_pos_embed = self.big_category_embedding[t_pos]
            t_neg_embed = self.big_category_embedding[t_neg]
        elif relation == 4:  # "big-category_small-category":
            h_embed = self.big_category_embedding[h]
            t_pos_embed = self.small_category_embedding[t_pos]
            t_neg_embed = self.small_category_embedding[t_neg]

        return h_embed, t_pos_embed, t_neg_embed

    def _cal_kg_loss_batch(self, city_id, h, t_pos, t_neg, relation):
        # TODO check whether this is right.
        h_embed, t_pos_embed, t_neg_embed = self._get_graph_embedding_for_kg_transR(city_id, h, t_pos, t_neg, relation)
        r_embed = self.graph_relation_embed[relation]
        W_r = self.graph_W_R[relation]

        # h_embed     (batch size, embedding dim)
        # t_pos_embed (batch size, embedding dim)
        # t_neg_embed (batch size, embedding dim)
        # r_embed     (relation dim)
        # W_r         (embedding dim, relation dim)

        r_mul_h = torch.matmul(h_embed, W_r)        # (batch size,relation dim)
        r_mul_pos = torch.matmul(t_pos_embed, W_r)  # (batch size,relation dim)
        r_mul_neg = torch.matmul(t_neg_embed, W_r)  # (batch size,relation dim)

        g1 = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos, 2), dim=1)
        g2 = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg, 2), dim=1)

        kg_loss = (-1.0) * nn.functional.logsigmoid(g2 - g1)
        kg_loss = torch.sum(kg_loss)
        return kg_loss

    def forward(self, *inputs):
        return self._cal_kg_loss_batch(*inputs)


class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(Aggregator, self).__init__()
        self.w1 = nn.Linear(in_dim, out_dim)
        self.w2 = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

    def forward(self, g, small_category_id, grid_id):
        funcs = dict()
        for e_type in g.etypes:
            funcs[e_type] = (dgl.function.u_mul_e('v', 'att', 'size'), dgl.function.sum('size', 'N_h'))
        g.multi_update_all(funcs, 'sum')

        grid_out_1 = self.activation(self.w1(g.nodes['grid'].data['v'] + g.nodes['grid'].data['N_h']))
        grid_out_2 = self.activation(self.w1(g.nodes['grid'].data['v'] * g.nodes['grid'].data['N_h']))
        grid_out = grid_out_1 + grid_out_2
        grid_out = self.message_dropout(grid_out)
        g.nodes['grid'].data['v'] = grid_out

        small_category_1 = self.activation(self.w1(g.nodes['small-category'].data['v'] +
                                             g.nodes['small-category'].data['N_h']))
        small_category_2 = self.activation(self.w1(g.nodes['small-category'].data['v'] *
                                             g.nodes['small-category'].data['N_h']))
        small_category_out = small_category_1 + small_category_2
        small_category_out = self.message_dropout(small_category_out)
        g.nodes['small-category'].data['v'] = small_category_out

        if args.use_category_ontology_diagram:
            big_cate_out1 = self.activation(self.w1(g.nodes['big-category'].data['v'] +
                                                    g.nodes['big-category'].data['N_h']))
            big_cate_out2 = self.activation(self.w1(g.nodes['big-category'].data['v'] +
                                                    g.nodes['big-category'].data['N_h']))
            big_cate_out = big_cate_out1 + big_cate_out2
            big_cate_out = self.message_dropout(big_cate_out)
            g.nodes['big-cate'].data['v'] = big_cate_out

        return small_category_out[small_category_id], grid_out[grid_id]


class KGMTRS(nn.Module):
    def __init__(self, n_big_category, n_small_category, n_city_grid, n_kg_relation,
                 graph_entity_relation_to_ID):
        super().__init__()
        self.graph_entity_relation_to_ID = graph_entity_relation_to_ID

        # knowledge graph
        self.knowledge_graph = KnowledgeGraph(n_big_category, n_small_category, n_city_grid, n_kg_relation)

    def forward(self, mode, *inputs):
        if mode == "cal_KG_transR":
            return self.knowledge_graph(*inputs)
        # elif mode == "cal_KG_attention":
        #     return self._compute_KG_attention(*inputs)
        # elif mode == "cal_NCF_loss" or mode == "test":
        #     return self._predict_and_compute_NCF_loss(*inputs)
        else:
            return None
