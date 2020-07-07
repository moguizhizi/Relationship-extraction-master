import tensorflow as tf
import numpy as np
import datetime
import network
import os
import sys
from sklearn.metrics import average_precision_score
from base.helper.args import get_args_parser
from base.helper.constant import PATH_NAME_PREFIX
from base.helper.constant import MODEL_DIR
from base.helper.constant import MODEL_ID_FILE
from base.helper.common import get_relation_num
from fnmatch import fnmatch

FLAGS = tf.app.flags.FLAGS


def save_parameter(model_id):
    f = open(MODEL_ID_FILE, 'w', encoding='utf-8')
    f.write(str(model_id))
    f.close()


def main(_):
    args = get_args_parser()

    wordembedding = np.load('./data/vec.npy',allow_pickle=True)

    test_settings = network.Settings()
    test_settings.vocab_size = len(wordembedding)
    test_settings.num_classes = get_relation_num(args.rel_id_file)
    test_settings.big_num = args.batch_size
    test_settings.regularizer = args.regularizer
    test_settings.num_steps = args.max_sentence_len

    if args.cell == 'gru':
        test_settings.cell_type = network.RNN__CELL_TYPE.GRU
    elif args.cell == 'lstm':
        test_settings.cell_type = network.RNN__CELL_TYPE.LSTM
    else:
        print("rnn cell type is error")
        sys.exit()

    precision = {}

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            with tf.variable_scope("model"):
                mtest = network.RNN_MODEL(is_training=False, word_embeddings=wordembedding, settings=test_settings, session=sess)
                mtest.construct_model()

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)

            testlist = get_parameter()
            
            for model_iter in testlist:
                # for compatibility purposes only, name key changes from tf 0.x to 1.x, compat_layer
                saver.restore(sess, PATH_NAME_PREFIX + "-" + str(model_iter))

                time_str = datetime.datetime.now().isoformat()
                print(time_str)
                print('Evaluating all test data and save data for PR curve')

                test_y = np.load('./data/testall_y.npy',allow_pickle=True)
                test_word = np.load('./data/testall_word.npy',allow_pickle=True)
                test_pos1 = np.load('./data/testall_pos1.npy',allow_pickle=True)
                test_pos2 = np.load('./data/testall_pos2.npy',allow_pickle=True)
                allprob = []
                acc = []
                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    prob, accuracy = mtest.process(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                                   test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                                   test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                                   test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                    prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
                    for single_prob in prob:
                        allprob.append(single_prob[1:])
                allprob = np.reshape(np.array(allprob), (-1))
                order = np.argsort(-allprob)

                print('saving all test result...')
                current_step = model_iter

                np.save('./out/allprob_iter_' + str(current_step) + '.npy', allprob)
                temp_allans = np.load('./data/allans.npy',allow_pickle=True)

                allans = temp_allans[0:allprob.shape[0]]

                # caculate the pr curve area
                average_precision = average_precision_score(allans, allprob)
                print(str(model_iter) + ' ' + 'PR curve area:' + str(average_precision))
                precision[model_iter]=average_precision

    model_id = max(precision,key=precision.get)

    save_parameter(model_id)

def get_parameter():

    metafiles = [name for name in os.listdir(MODEL_DIR)
                 if fnmatch(name, '*.meta')]

    files = []
    for filename in metafiles:
        files.append(filename.split('.')[0])

    parameter_list = []
    for file in files:
        parameter_list.append(file.split('-')[1])

    return parameter_list

if __name__ == "__main__":
    tf.app.run()
