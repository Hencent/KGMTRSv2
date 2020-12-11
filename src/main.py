#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

import torch
from time import time
from src.src.data_loader import DataLoader
from src.src.system_init import system_init
from src.src.log_tool import logging
from src.src.KGMTRS import KGMTRS

from src.args import args

CUDA_AVAILABLE = False
DEVICE = None
N_GPU = 0


if __name__ == '__main__':
    # system init and CUDA
    if args.use_category_ontology_diagram:
        print("Using category ontology diagram")
    else:
        print("Not using category ontology diagram")

    system_init()

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(args.seed)
        N_GPU = torch.cuda.device_count()
    DEVICE = torch.device("cuda:{}".format(0) if CUDA_AVAILABLE else "cpu")
    logging.info("|--           system init done.")

    data = DataLoader(DEVICE)
    logging.info("|--           load data done.")

    # model and optimizer
    logging.info("|--           build model and optimizer.")
    model = KGMTRS(data.n_big_category, data.n_small_category,
                   data.n_city_grid, data.n_kg_relation, data.graph_entity_relation_to_ID)
    if CUDA_AVAILABLE:
        model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(model.parameters())

    # train model
    logging.info("|--           begin training model.")
    kg_loss_list = []
    ncf_loss_list = []
    ndcg_list = []
    hr_list = []

    for epoch in range(args.n_epoch):
        data.shuffle_train_data_index()
        epoch_start_time = time()
        model.train()

        # train kg transR
        time_kg_transR_start = time()
        for city_id, city_name in enumerate(args.city_list):
            for kg_iter in range(len(data.train_batch_ncf_index[city_id])):
                relation, head, pos_tail, neg_tail = \
                    data.get_train_keg_batch_data(city_id, data.train_batch_ncf_index[city_id][kg_iter])
                optimizer.zero_grad()
                kg_iter_loss = model("cal_KG_transR", city_id, head, pos_tail, neg_tail, relation)
                kg_iter_loss.backward()
                optimizer.step()

                if kg_iter % args.print_iter_frequency_kg == 0:
                    logging.info(
                        '|--            Epoch {:04d} | KG TransR Training {}: | Iter {:05d} / {:05d} '
                        '| Iter Loss {:.4f}'.format(epoch, city_name, kg_iter,
                                                    len(data.train_batch_ncf_index[city_id]),
                                                    kg_iter_loss.cpu().item()))

        logging.info('|--            Epoch {:04d} | KG TransR Done | Total Time {:.1f}s'.
                     format(epoch, time() - time_kg_transR_start))

        # train ncf
