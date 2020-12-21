#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : preprocess_tool.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2020/12/5
# @Desc  : None

from src.preprocess.extract_data import data_extract_and_generate_test_data
from src.preprocess.build_graph import build_city_graph

from src.args import args

if __name__ == "__main__":
    print("|--    begin extract data and generate test data.")
    # data_extract_and_generate_test_data()

    print("|--    build city graph.")
    build_city_graph()

    print("|--    preprocess data done.")
