import tensorflow as tf
import numpy as np
import datetime
import network
import os
import shutil
import sys

from base.helper.args import get_args_parser
from base.helper.constant import PATH_NAME_PREFIX, MODEL_DIR
from base.helper.constant import MAX_SENTENCE_LENGTH
from base.helper.common import get_relation_num

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')


def main(_):

    if os.path.isdir(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    args = get_args_parser()

    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy',allow_pickle=True)

    print('reading training data')
    train_y = np.load('./data/train_y.npy',allow_pickle=True)
    train_word = np.load('./data/train_word.npy',allow_pickle=True)
    train_pos1 = np.load('./data/train_pos1.npy',allow_pickle=True)
    train_pos2 = np.load('./data/train_pos2.npy',allow_pickle=True)
    train_entitis = np.load('./data/train_entities.npy',allow_pickle=True)

    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = get_relation_num(args.rel_id_file)

    settings.big_num = args.batch_size
    settings.num_steps = args.max_sentence_len
    settings.regularizer =args.regularizer
    settings.num_epochs = args.num_train_epochs
    settings.entities_len = args.max_entities_len

    if args.cell == 'gru':
        settings.cell_type = network.RNN__CELL_TYPE.GRU
    elif args.cell == 'lstm':
        settings.cell_type = network.RNN__CELL_TYPE.LSTM
    else:
        print("rnn cell type is error")
        sys.exit()

    if args.weight == 'normal':
        settings.weight_type = network.WEIGHT_TYPE.NORMAL
    elif args.weight == 'relation':
        settings.weight_type = network.WEIGHT_TYPE.RELATION
    else:
        print("weight_type is error")
        sys.exit()

    big_num = settings.big_num

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = network.RNN_MODEL(is_training=True, word_embeddings=wordembedding, settings=settings, session=sess)
                m.construct_model()
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(m.settings.lr)

            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
           
            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, entitis_batch, big_num):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                total_entities = []
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                    for entitis in entitis_batch[i]:
                        total_entities.append(entitis)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)
                total_entities = np.array(total_entities)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch
                feed_dict[m.input_entities] = total_entities

                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)

                if step % 50 == 0:
                    tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                    print(tempstr)

            for one_epoch in range(settings.num_epochs):

                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)

                for i in range(int(len(temp_order) / float(settings.big_num))):

                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []
                    temp_entitis = []

                    temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                        temp_entitis.append(train_entitis[k])
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > MAX_SENTENCE_LENGTH:
                        print('out of range')
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)
                    temp_entitis = np.array(temp_entitis)

                    train_step(temp_word, temp_pos1, temp_pos2, temp_y, temp_entitis, settings.big_num)

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step >= args.begin_save_steps and current_step % 100 == 0:
                        print('saving model')
                        path = saver.save(sess, PATH_NAME_PREFIX, global_step=current_step)
                        tempstr = 'have saved model to ' + path
                        print(tempstr)


if __name__ == "__main__":
    tf.app.run()
