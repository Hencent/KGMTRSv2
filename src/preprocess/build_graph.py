#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : build_graph.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

import csv
import os
import pandas as pd
import dgl
from src.tool.shared_tool import load_category_relation, load_category

from src.args import args


class _SingleCityGraphBuilder(object):
    def __init__(self, city_id):
        self.city_id = city_id

        # load necessary info in order to build graph
        print("| | |--       load necessary information in order to build graph.")
        self.category_grid_relation, self.grid_relation = self._load_graph_relation()
        self.category_relation = load_category_relation()
        _, _, n_big_category, n_small_category = load_category()
        n_grid = self._get_n_grid()

        # build graph
        print("| | |--       build graph.")
        self.graph = self._build_graph(n_big_category, n_small_category, n_grid)

        # save handled city data
        print("| | |--       save city graph data.")
        self._save_graph()

    def _get_n_grid(self):
        data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[self.city_id])
        with open(os.path.join(data_dir, "base_info.csv"), 'r') as f:
            next(f)
            next(f)
            reader = csv.reader(f, delimiter=',')
            n_grid = [int(row[1]) for row in reader][0]

        return n_grid

    def _load_graph_relation(self):
        data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[self.city_id])

        category_grid_relation = pd.read_csv(os.path.join(data_dir, "category_grid_relation.csv"))
        grid_relation = pd.read_csv(os.path.join(data_dir, "grid_relations.csv"))

        return category_grid_relation, grid_relation

    def _build_graph(self, n_big_category, n_small_category, n_grid):
        graph_data = {
            ('small-category', 'small-category_grid', 'grid'): (self.category_grid_relation["small category ID"],
                                                                self.category_grid_relation["grid ID"]),
            ('grid', 'grid_small-category', 'small-category'): (self.category_grid_relation["grid ID"],
                                                                self.category_grid_relation["small category ID"]),

            ('grid', 'grid_grid', 'grid'): (self.grid_relation["grid source ID"],
                                            self.grid_relation["grid target ID"]),
        }
        num_nodes_dict = {'small-category': n_small_category, 'grid': n_grid}

        if args.use_category_ontology_diagram:
            graph_data.update({
                ('small-category', 'small-category_big-big-category', 'big-category'): (
                    self.category_relation["small_cate"], self.category_relation["big_cate"]),
                ('big-category', 'big-category_small-category', 'small-category'): (
                    self.category_relation["big_cate"], self.category_relation["small_cate"]),
            })
            num_nodes_dict.update({
                'big-category': n_big_category,
            })

        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        return g

    def _save_graph(self):
        data_dir = os.path.join(args.preprocessed_data_dir, args.city_list[self.city_id])
        dgl.save_graphs(os.path.join(data_dir, "city_graph"), [self.graph])


def build_city_graph():
    for city_id, city_name in enumerate(args.city_list):
        print("| |--         handle city graph: {}.".format(city_name))
        _SingleCityGraphBuilder(city_id)

