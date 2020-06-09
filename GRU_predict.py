from pprint import pprint

import tensorflow as tf
import numpy as np
import network
import codecs
import pandas as pd
import os

from base.helper.constant import MODEL_ID_FILE
from base.helper.constant import PATH_NAME_PREFIX
from base.helper.common import get_id_relation
from base.helper.common import get_relation_id
from base.helper.common import get_word_id
from base.helper.args import get_args_parser
from base.helper.common import get_relation_num


FLAGS = tf.app.flags.FLAGS

def get_parameter(MODEL_ID_FILE):
    f = open(MODEL_ID_FILE, 'r', encoding='utf-8')
    content = f.readline()
    content = content.strip()
    f.close()
    return content

def write_to_file(file,entity1, entity2, relation):

    frame_list = []
    temp_list = []
    temp_list.append(entity1)
    temp_list.append(entity2)
    temp_list.append(relation)
    frame_list.append(temp_list)

    df = pd.DataFrame(frame_list, columns=list('ABC'))
    df.to_csv(file, index=False, header=False, sep=",", encoding="utf_8_sig", mode="a")


def main(_):

    args = get_args_parser()

    model_id = get_parameter(MODEL_ID_FILE)

    pathname = PATH_NAME_PREFIX + "-" + str(model_id)
    
    wordembedding = np.load('./data/vec.npy')
    predict_settings = network.Settings()
    predict_settings.vocab_size = len(wordembedding)
    predict_settings.num_classes = get_relation_num(args.rel_id_file)
    predict_settings.big_num = 1
    predict_settings.num_steps = args.max_sentence_len
    predict_settings.regularizer = args.regularizer
    
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            with tf.variable_scope("model"):
                mpredict = network.GRU(is_training=False, word_embeddings=wordembedding, settings=predict_settings, session=sess)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)
            saver.restore(sess, pathname)

            word2id = get_word_id(args.vec_file)

            relation2id = get_relation_id(args.rel_id_file)

            id2relation = get_id_relation(args.rel_id_file)

            input_data = codecs.open(args.pre_file, 'r', 'utf-8')

            # 删除预测的结果文件
            if os.path.exists(args.result_file):
                os.remove(args.result_file)

            for line in input_data.readlines():

                line = line.strip()
                pos_embedding = mpredict.get_pos_embedding(line, word2id)
                batch = mpredict.get_batch(pos_embedding, relation2id)
                prob, accuracy = mpredict.process(batch.word_batch, batch.pos1_batch, batch.pos2_batch,
                                                  batch.relation_batch)

                prob = np.reshape(np.array(prob), (1, predict_settings.num_classes))[0]
                top1_id = prob.argsort()[-1:]
                rel_id = top1_id[0]

                en1, en2, _ = line.strip().split()
                write_to_file(args.result_file, en1, en2, id2relation[rel_id])

            input_data.close()

if __name__ == "__main__":
    tf.app.run()
