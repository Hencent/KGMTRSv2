#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : KGMTRS.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/10
# @Desc  : None

import torch
import torch.nn as nn
import numpy as np
import dgl
from src.tool.metrics import hit_ratio_at_K, ndcg_at_K

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

    def get_node_embedding(self, node_id_list, city_id, node_type):
        if node_type == "grid":
            return self.city_grid_embedding[city_id][node_id_list]
        elif node_type == "big-category":
            return self.big_category_embedding[node_id_list]
        elif node_type == "small-category":
            return self.small_category_embedding[node_id_list]
        else:
            return None


class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(Aggregator, self).__init__()
        self.w1 = nn.Linear(in_dim, out_dim)
        self.w2 = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

    def forward(self, g, category_id_list, pos_grid_id, neg_grid_id, stage=0):
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
            g.nodes['big-category'].data['v'] = big_cate_out

        if stage == 0:
            # return [g.nodes['small-category'].data['v'][cats] for cats in category_id_list], \
            #     grid_out[pos_grid_id], grid_out[neg_grid_id]
            return [small_category_out[cats] for cats in category_id_list], \
                grid_out[pos_grid_id], grid_out[neg_grid_id]
        else:
            return small_category_out[category_id_list], grid_out[pos_grid_id]


class NCF(nn.Module):
    def __init__(self):
        super().__init__()
        NCF_dim = (sum(args.conv_dim_list) + args.embedding_size) * 2 + args.city_bias_size

        self.MLP = nn.Sequential()
        dim_list = [NCF_dim] + args.ncf_dim_list
        for i in range(1, len(dim_list)):
            self.MLP.add_module("Linear_{}".format(i), nn.Linear(dim_list[i-1], dim_list[i]))
            self.MLP.add_module("LeakyRelu_{}".format(i), nn.LeakyReLU())

    def forward(self, in_data):
        return self.MLP(in_data)


class KGMTRS(nn.Module):
    def __init__(self, n_big_category, n_small_category, n_city_grid, n_kg_relation,
                 graph_entity_relation_to_ID):
        super().__init__()
        self.graph_entity_relation_to_ID = graph_entity_relation_to_ID

        # knowledge graph
        self.knowledge_graph = KnowledgeGraph(n_big_category, n_small_category, n_city_grid, n_kg_relation)

        # graph aggregation layers
        self.aggregation_layers = nn.ModuleList()
        conv_dim_list = [args.embedding_size] + args.conv_dim_list
        for idx, _ in enumerate(args.conv_dim_list):
            self.aggregation_layers.append(Aggregator(conv_dim_list[idx], conv_dim_list[idx + 1],
                                                      args.mess_dropout[idx]))

        # NCF
        self.NCF = NCF()

    def _calculate_KG_attention(self, city_id, g):
        relation_name = ""
        city_id = city_id

        def _KG_attention(edges):
            relation_id = self.graph_entity_relation_to_ID[relation_name]
            W_r = self.knowledge_graph.graph_W_R[relation_id]
            r_embed = self.knowledge_graph.graph_relation_embed[relation_id]  # (1, relation_dim)
            node_types = relation_name.split('_')
            h_node_ids, t_node_ids, _ = edges.edges()

            h_nodes_embeddings = self.knowledge_graph.get_node_embedding(h_node_ids, city_id, node_types[0])
            t_nodes_embeddings = self.knowledge_graph.get_node_embedding(t_node_ids, city_id, node_types[1])
            r_mul_h = torch.matmul(h_nodes_embeddings, W_r)  # (n_edge, relation_dim)
            r_mul_t = torch.matmul(t_nodes_embeddings, W_r)  # (n_edge, relation_dim)

            att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)  # (n_edge, 1)
            return {'att': att}

        def _edge_soft_max(graph, score):
            with graph.local_scope():
                funcs = dict()
                for k, v in score.items():
                    _, e_type, _ = k
                    graph.edges[e_type].data['Wh'] = torch.exp(v)
                    funcs[e_type] = (dgl.function.copy_e('Wh', 'm'), dgl.function.sum('m', 'h'))
                graph.multi_update_all(funcs, 'sum')

                for e_name in g.etypes:
                    g.apply_edges(dgl.function.e_div_v('Wh', 'h', 'out'), etype=e_name)
                return g.edata.pop('out')

        with g.local_scope():
            for type_name in g.etypes:
                relation_name = type_name
                g.apply_edges(_KG_attention, etype=type_name)
            return _edge_soft_max(g, g.edata.pop('att'))

    def _propagation_for_NCF(self, city_id, g, cate_list, pos_grid_list, neg_grid_list):
        with g.local_scope():

            g.nodes['small-category'].data['v'] = self.knowledge_graph.small_category_embedding
            g.nodes['grid'].data['v'] = self.knowledge_graph.city_grid_embedding[city_id]
            if args.use_category_ontology_diagram:
                g.nodes['big-category'].data['v'] = self.knowledge_graph.big_category_embedding

            cate_list_embedding = [[g.nodes['small-category'].data['v'][cats]] for cats in cate_list]
            pos_grid_embedding = [g.nodes['grid'].data['v'][pos_grid_list]]
            neg_grid_embedding = [g.nodes['grid'].data['v'][neg_grid_list]]

            for layer in self.aggregation_layers:
                cate_meta, pos_grid_meta, neg_grid_meta = layer(g, cate_list, pos_grid_list, neg_grid_list)
                for k, v in enumerate(cate_list_embedding):
                    v.append(cate_meta[k])
                pos_grid_embedding.append(pos_grid_meta)
                neg_grid_embedding.append(neg_grid_meta)

            cate_list_embedding = [torch.cat(single_cat, dim=1) for single_cat in cate_list_embedding]
            pos_grid_embedding = torch.cat(pos_grid_embedding, dim=1)
            neg_grid_embedding = torch.cat(neg_grid_embedding, dim=1)

        return cate_list_embedding, pos_grid_embedding, neg_grid_embedding

    @staticmethod
    def _merge_cate_list_for_NCF(cate_list_embedding, pos_gird_embedding, neg_grid_embedding):
        pos_merged_embedding = []
        neg_merged_embedding = []
        for idx, cats in enumerate(cate_list_embedding):
            if cats.shape[0] < 2:
                pos_merged_embedding.append(cats[0])
                neg_merged_embedding.append(cats[0])
            else:
                # TODO check
                c_mul_pos_g = torch.exp(torch.matmul(pos_gird_embedding[idx], cats.T))
                c_mul_neg_g = torch.exp(torch.matmul(neg_grid_embedding[idx], cats.T))
                c_pos_g_att = c_mul_pos_g / torch.sum(c_mul_pos_g, dim=1).unsqueeze(1)
                c_neg_g_att = c_mul_neg_g / torch.sum(c_mul_neg_g, dim=1).unsqueeze(1)
                single_merged_pos = torch.matmul(c_pos_g_att, cats.T)
                single_merged_neg = torch.matmul(c_neg_g_att, cats.T)
                pos_merged_embedding.append(single_merged_pos)
                neg_merged_embedding.append(single_merged_neg)

        return torch.stack(pos_merged_embedding), torch.stack(neg_merged_embedding)

    def _calculate_NCF_loss(self, city_id, g, cate_list, pos_grid_list, neg_grid_list):
        cate_list_embedding, pos_gird_embedding, neg_grid_embedding = \
            self._propagation_for_NCF(city_id, g, cate_list, pos_grid_list, neg_grid_list)
        cate_pos_embedding, cate_neg_embedding = \
            self._merge_cate_list_for_NCF(cate_list_embedding, pos_gird_embedding, neg_grid_embedding)

        city_bias = self.knowledge_graph.city_bias[city_id].repeat(len(cate_list), 1)

        pos_score = self.NCF(torch.cat((cate_pos_embedding, pos_gird_embedding, city_bias), 1)).squeeze()
        neg_score = self.NCF(torch.cat((cate_neg_embedding, neg_grid_embedding, city_bias), 1)).squeeze()

        ncf_loss = (-1.0) * nn.functional.logsigmoid(pos_score - neg_score)
        ncf_loss = torch.sum(ncf_loss)

        return ncf_loss

    @staticmethod
    def _merge_cate_for_test(cats, grids):
        c_mul_g = torch.exp(torch.matmul(grids, cats.T))
        c_g_att = c_mul_g / torch.sum(c_mul_g, dim=2).unsqueeze(2)
        merged_embedding = torch.matmul(c_g_att, cats)
        return merged_embedding

    def _propagation_for_test(self, g, cats, grids):
        with g.local_scope():
            g.nodes['small-category'].data['v'] = self.knowledge_graph.small_category_embedding
            g.nodes['grid'].data['v'] = self.knowledge_graph.city_grid_embedding[args.target_city_id]
            if args.use_category_ontology_diagram:
                g.nodes['big-category'].data['v'] = self.knowledge_graph.big_category_embedding

            test_len, single_test_len = grids.shape
            grids = grids.reshape(1, -1).squeeze()

            cate_list_embedding = [g.nodes['small-category'].data['v'][cats]]
            grids_embedding = [g.nodes['grid'].data['v'][grids]]

            for layer in self.aggregation_layers:
                cate_meta, grids_meta = layer(g, cats, grids, None, 1)
                cate_list_embedding.append(cate_meta)
                grids_embedding.append(grids_meta)

            cate_list_embedding = torch.cat(cate_list_embedding, dim=1)
            grids_embedding = torch.cat(grids_embedding, dim=1).reshape(test_len, single_test_len, -1)

        return cate_list_embedding, grids_embedding

    def _test(self, city_graph, target_cate_ids, test_grids):
        cate_embedding, grid_embedding = self._propagation_for_test(city_graph, target_cate_ids, test_grids)
        merged_cate_embedding = self._merge_cate_for_test(cate_embedding, grid_embedding)
        city_bias = self.knowledge_graph.city_bias[args.target_city_id].repeat(len(test_grids[0]), 1)

        hr_list = []
        ndcg_list = []
        for test_idx in range(len(test_grids)):
            ncf_input = torch.cat((merged_cate_embedding[test_idx], grid_embedding[test_idx], city_bias), 1)
            score = self.NCF(ncf_input).squeeze()
            _, sorted_indices = torch.sort(score, descending=True)

            hr_list.append(hit_ratio_at_K(0, sorted_indices[:args.K], args.K))
            ndcg_list.append(ndcg_at_K(0, sorted_indices[:args.K], args.K))
        return np.mean(hr_list), np.mean(ndcg_list)

    def forward(self, mode, *inputs):
        if mode == "cal_KG_transR":
            return self.knowledge_graph(*inputs)
        elif mode == "cal_NCF_loss":
            return self._calculate_NCF_loss(*inputs)
        elif mode == "cal_KG_attention":
            return self._calculate_KG_attention(*inputs)
        elif mode == "test":
            return self._test(*inputs)
        else:
            return None
