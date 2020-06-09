# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/5/26 17:37
@file: constant.py
@desc: 
"""

import os

root_path = os.path.abspath(".")

"数据集文件夹"
ORIGIN_DIR = os.path.join(root_path, 'origin_data')

"模型保文件夹路径"
MODEL_DIR = os.path.join(root_path, 'model')

"out文件夹路径"
OUT_DIR = os.path.join(root_path, 'out')

"模型保文件前缀"
MODEL_FILE_PREFIX = "ATT_GRU_model"

"模型文件路径前缀"
PATH_NAME_PREFIX = os.path.join(MODEL_DIR, MODEL_FILE_PREFIX)

"模型Id文件"
MODEL_ID_FILE = os.path.join('data', 'model_id.txt')

"语句最大长度"
MAX_SENTENCE_LENGTH = 1500