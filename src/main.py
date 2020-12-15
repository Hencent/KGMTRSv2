#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

import torch
from time import time
import matplotlib.pyplot as plt
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train model
    logging.info("|--           begin training model.")
    kg_loss_list = []
    ncf_loss_list = []
    # hr_list = []
    # ndcg_list = []
    hr_list_list = [[] for k in args.K_list]
    ndcg_list_list = [[] for k in args.K_list]

    for epoch in range(args.n_epoch):
        # data.shuffle_train_data_index()
        epoch_start_time = time()
        model.train()

        # train kg transR
        kg_epoch_loss_total = 0
        kg_iter_sum = 0
        time_kg_transR_start = time()

        for city_id, city_name in enumerate(args.city_list):
            kg_iter_sum += len(data.train_batch_ncf_index[city_id])
            for kg_iter in range(len(data.train_batch_ncf_index[city_id])):
                relation, head, pos_tail, neg_tail = \
                    data.get_train_keg_batch_data(city_id, data.train_batch_ncf_index[city_id][kg_iter])
                optimizer.zero_grad()
                kg_iter_loss = model("cal_KG_transR", city_id, head, pos_tail, neg_tail, relation)
                kg_iter_loss.backward()
                optimizer.step()

                kg_epoch_loss_total += kg_iter_loss.cpu().item()

                if kg_iter % args.print_iter_frequency_kg == 0:
                    logging.info(
                        '|--            Epoch {:03d} | KG TransR Training {}: | Iter {:04d} / {:04d} '
                        '| Iter Loss {:.4f}'.format(epoch, city_name, kg_iter,
                                                    len(data.train_batch_ncf_index[city_id]),
                                                    kg_iter_loss.cpu().item()))

        kg_epoch_mean = kg_epoch_loss_total / kg_iter_sum
        kg_loss_list.append(kg_epoch_mean)

        logging.info('|--            Epoch {:03d} | KG TransR Done | Total Time {:.1f}s | Mean Loss {:.4f}'.
                     format(epoch, time() - time_kg_transR_start, kg_epoch_mean))

        # calculate attention for propagation
        for city_id, _ in enumerate(args.city_list):
            with torch.no_grad():
                attention_score = model("cal_KG_attention", city_id, data.city_graphs[city_id])
            for k, v in attention_score.items():
                data.city_graphs[city_id].edges[k[1]].data['att'] = v

        # train ncf
        ncf_epoch_loss_total = 0
        ncf_iter_sum = 0
        time_ncf_transR_start = time()

        for city_id, city_name in enumerate(args.city_list):
            ncf_iter_sum += len(data.train_batch_ncf_index[city_id])
            for ncf_iter in range(len(data.train_batch_ncf_index[city_id])):
                cate_list, pos_grid_list, neg_grid_list = \
                    data.get_train_ncf_batch_data(city_id, data.train_batch_ncf_index[city_id][ncf_iter])
                optimizer.zero_grad()
                ncf_iter_loss = model("cal_NCF_loss", city_id, data.city_graphs[city_id],
                                      cate_list, pos_grid_list, neg_grid_list)
                ncf_iter_loss.backward()
                optimizer.step()

                ncf_epoch_loss_total += ncf_iter_loss.cpu().item()

                if ncf_iter % args.print_iter_frequency_kg == 0:
                    logging.info(
                        '|--            Epoch {:03d} | KG NCF Training {}: | Iter {:04d} / {:04d} '
                        '| Iter Loss {:.4f}'.format(epoch, city_name, ncf_iter,
                                                    len(data.train_batch_ncf_index[city_id]),
                                                    ncf_iter_loss.cpu().item()))

        ncf_epoch_mean = ncf_epoch_loss_total / ncf_iter_sum
        ncf_loss_list.append(ncf_epoch_mean)

        logging.info('|--            Epoch {:03d} | NCF Done | Total Time {:.1f}s | Mean Loss {:.4f}'.
                     format(epoch, time() - time_ncf_transR_start, ncf_epoch_mean))

        # test
        model.eval()
        with torch.no_grad():
            test_grids, target_cate_ids = data.get_test_data()
            # hr, ndcg = model("test", data.city_graphs[args.target_city_id], target_cate_ids, test_grids)
            # hr_list.append(hr)
            # ndcg_list.append(ndcg)
            # logging.info('|--            Epoch {:03d} | Test : | HR@{} {:.4f} | NDCG@{} {:.4f}'.
            #              format(epoch, args.K, hr, args.K, ndcg))

            hr_list_val, ndcg_list_val = \
                model("test", data.city_graphs[args.target_city_id], target_cate_ids, test_grids)

            for k_index, _ in enumerate(args.K_list):
                hr_list_list[k_index].append(hr_list_val[k_index])
                ndcg_list_list[k_index].append(ndcg_list_val[k_index])

            test_base_logging_info = '|--            Epoch {:03d} | Test :'.format(epoch)
            test_metrics_logging_info_list = [' | HR@{} {:.4f} - NDCG@{} {:.4f}'.format(k_val, hr_list_val[k_idx],
                                                                                        k_val, ndcg_list_val[k_idx])
                                              for k_idx, k_val in enumerate(args.K_list)]
            test_metrics_logging_info_list = ''.join(test_metrics_logging_info_list)

            logging.info(test_base_logging_info + test_metrics_logging_info_list)

    # train model done.
    logging.info("|--           training model done.")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel("epoch")
    plt.ylabel("kg loss every epoch")
    plt.plot(range(len(kg_loss_list)), kg_loss_list)
    plt.subplot(1, 2, 2)
    plt.xlabel("epoch")
    plt.ylabel("ncf loss every epoch")
    plt.plot(range(len(ncf_loss_list)), ncf_loss_list)
    plt.savefig('loss.png')
    plt.show()

    # plt.figure(2)
    # plt.subplot(1, 2, 1)
    # plt.xlabel("epoch")
    # plt.ylabel("HR@{}".format(args.K))
    # plt.plot(range(len(hr_list)), hr_list)
    # plt.subplot(1, 2, 2)
    # plt.xlabel("epoch")
    # plt.ylabel("NDCG@{}".format(args.K))
    # plt.plot(range(len(ndcg_list)), ndcg_list)
    # plt.savefig('metrics.png')
    # plt.show()

    for k_idx, k_val in enumerate(args.K_list):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.xlabel("epoch")
        plt.ylabel("HR@{}".format(k_val))
        plt.plot(range(len(hr_list_list[k_index])), hr_list_list[k_index])
        plt.subplot(1, 2, 2)
        plt.xlabel("epoch")
        plt.ylabel("NDCG@{}".format(k_val))
        plt.plot(range(len(ndcg_list_list[k_index])), ndcg_list_list[k_index])
        plt.savefig('metrics@{}.png'.format(k_val))
        plt.show()
