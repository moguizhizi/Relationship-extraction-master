
import tensorflow as tf

import data_util
from base.helper.args import get_args_parser



def main(_):
    args = get_args_parser()
    label_map = data_util.update_label_map(args.rel_id_file)
    data_util.statistics('./origin_data/train.txt', label_map)
    data_util.statistics('./origin_data/test.txt', label_map)

    # 句子的最大长度
    max = data_util.get_max_sentence_length('./origin_data/train.txt')
    print("train.txt max sentence length:" + str(max))

    max = data_util.get_max_sentence_length('./origin_data/test.txt')
    print("test.txt max sentence length:" + str(max))

    # 句子的最小长度
    min = data_util.get_min_sentence_length('./origin_data/train.txt')
    print("train.txt max sentence length:" + str(min))

    min = data_util.get_min_sentence_length('./origin_data/test.txt')
    print("test.txt max sentence length:" + str(min))

    # 实体对的最大长度
    max = data_util.get_max_entities_length('./origin_data/train.txt')
    print("train.txt max entities length:" + str(max))

    # 句子长度占比
    print("sentences length info:")
    data_util.get_sen_proportion('./origin_data/train.txt', args.sen_sec_len)

    # 实体对占比
    print("entities length info:")
    data_util.get_entities_proportion('./origin_data/train.txt', args.entity_sec_len)

if __name__ == "__main__":
    tf.app.run()
