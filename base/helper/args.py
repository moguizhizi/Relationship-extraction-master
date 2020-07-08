# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/5/25 14:28
@file: args.py
@desc: 
"""

import argparse
import os

from base.helper.constant import OUT_DIR
from base.helper.constant import ORIGIN_DIR


__all__ = ['get_args_parser']


def get_args_parser():

    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group('File Paths',
                                       'config the path, checkpoint and filename of a pretrained')

    group1.add_argument('-rel_id_file', type=str, default=os.path.join(ORIGIN_DIR, 'relation2id.txt'))

    group1.add_argument('-vec_file', type=str, default=os.path.join(ORIGIN_DIR, 'vec.txt'))

    group1.add_argument('-pre_file', type=str, default=os.path.join(ORIGIN_DIR, 'pre.txt'),
                        help='predict file path')

    group1.add_argument('-result_file', type=str, default=os.path.join(OUT_DIR, 'result.csv'),
                        help='predict result file path')

    group2 = parser.add_argument_group('Model Config', 'config the model params')

    group2.add_argument('-batch_size', type=int, default=50,
                        help='Total batch size for training, test and predict.')

    group2.add_argument('-max_sentence_len', type=int, default=200,
                        help='Max the number of sentence word')

    group2.add_argument('-max_entities_len', type=int, default=70,
                        help='Max the number of entities word')

    group2.add_argument('-sen_sec_len', type=int, default=30,
                        help='the length of the section')

    group2.add_argument('-entity_sec_len', type=int, default=10,
                        help='the length of the section')

    group2.add_argument('-regularizer', type=float, default=0.0001,
                        help='Regularizer')

    group2.add_argument('-num_train_epochs', type=int, default=50,
                        help='Total number of training epochs to perform.')

    group2.add_argument('-begin_save_steps', type=int, default=4000,
                        help='Save model data from step')

    group2.add_argument('-cell', type=str, default='gru',
                        help='which rnn cell used.')




    return parser.parse_args()
