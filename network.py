import tensorflow as tf
import numpy as np
from enum import Enum

class Settings(object):
    def __init__(self):
        # 词向量的个数
        self.vocab_size = 16691
        self.num_steps = 70
        # 训练集的训练次数
        self.num_epochs = 10
        # 关系种类的总数
        self.num_classes = 5
        # 隐藏层的神经元数量
        self.hidden_unit = 230
        self.keep_prob = 0.5
        # 全连接层的层数
        self.num_layers = 1
        self.pos_size = 5
        # pos_num必须大于函数pos_embed的最大返回值
        self.pos_num = 123
        # the number of entity pairs of each batch during training or testing
        self.big_num = 50
        # 学习率
        self.lr = 0.0005
        # 正则项
        self.regularizer = 0.0001
        # RNN类型
        self.cell_type = RNN__CELL_TYPE.GRU


class RNN_MODEL:
    def __init__(self, is_training, word_embeddings, settings, session):

        self.session = session
        self.settings = settings

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_steps], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_steps], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_steps], name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.settings.num_classes], name='input_y')
        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[self.settings.big_num + 1], name='total_shape')
        self.total_num = self.total_shape[-1]

        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0

        self._initial_state_forward = None
        self._initial_state_backward = None

        self.word_embeddings = word_embeddings
        self.is_training = is_training

    def construct_model(self):

        word_embedding = tf.get_variable(initializer=self.word_embeddings, name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding', [self.settings.pos_num, self.settings.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [self.settings.pos_num, self.settings.pos_size])

        attention_w = tf.get_variable('attention_omega', [self.settings.hidden_unit, 1])
        sen_a = tf.get_variable('attention_A', [self.settings.hidden_unit])
        sen_r = tf.get_variable('query_r', [self.settings.hidden_unit, 1])
        relation_embedding = tf.get_variable('relation_embedding',
                                             [self.settings.num_classes, self.settings.hidden_unit])
        sen_d = tf.get_variable('bias_d', [self.settings.num_classes])

        rnn_cell_forward, rnn_cell_backward = self._bi_dir_rnn()

        cell_forward = tf.contrib.rnn.MultiRNNCell([rnn_cell_forward] * self.settings.num_layers)
        cell_backward = tf.contrib.rnn.MultiRNNCell([rnn_cell_backward] * self.settings.num_layers)

        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []

        self._initial_state_forward = cell_forward.zero_state(self.total_num, tf.float32)
        self._initial_state_backward = cell_backward.zero_state(self.total_num, tf.float32)

        # embedding layer
        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(word_embedding, self.input_word),
                                                   tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
                                                   tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)])

        inputs_backward = tf.concat(axis=2,
                                    values=[tf.nn.embedding_lookup(word_embedding, tf.reverse(self.input_word, [1])),
                                            tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1, [1])),
                                            tf.nn.embedding_lookup(pos2_embedding, tf.reverse(self.input_pos2, [1]))])

        outputs_forward = []

        state_forward = self._initial_state_forward

        with tf.variable_scope('RNN_FORWARD') as scope:
            for step in range(self.settings.num_steps):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_forward, state_forward) = cell_forward.call(inputs_forward[:, step, :], state_forward)
                outputs_forward.append(cell_output_forward)

        outputs_backward = []

        state_backward = self._initial_state_backward
        with tf.variable_scope('RNN_BACKWARD') as scope:
            for step in range(self.settings.num_steps):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_backward, state_backward) = cell_backward.call(inputs_backward[:, step, :], state_backward)
                outputs_backward.append(cell_output_backward)

        output_forward = tf.reshape(tf.concat(axis=1, values=outputs_forward),
                                    [self.total_num, self.settings.num_steps, self.settings.hidden_unit])
        output_backward = tf.reverse(tf.reshape(tf.concat(axis=1, values=outputs_backward),
                                                [self.total_num, self.settings.num_steps, self.settings.hidden_unit]), [1])

        # word-level attention layer
        output_h = tf.add(output_forward, output_backward)
        attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(
                tf.reshape(tf.tanh(output_h), [self.total_num * self.settings.num_steps, self.settings.hidden_unit]),
                attention_w),
                       [self.total_num, self.settings.num_steps])), [self.total_num, 1, self.settings.num_steps]), output_h),
            [self.total_num, self.settings.hidden_unit])

        # sentence-level attention layer
        for i in range(self.settings.big_num):

            sen_repre.append(tf.tanh(attention_r[self.total_shape[i]:self.total_shape[i + 1]]))
            batch_size = self.total_shape[i + 1] - self.total_shape[i]

            sen_alpha.append(
                tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre[i], sen_a), sen_r), [batch_size])),
                           [1, batch_size]))

            sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [self.settings.hidden_unit, 1]))
            sen_out.append(
                tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s[i]), [self.settings.num_classes]), sen_d))

            self.prob.append(tf.nn.softmax(sen_out[i]))

            with tf.name_scope("output"):
                self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

            with tf.name_scope("loss"):

                self.temp_loss = tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.input_y[i])
                self.temp_mean = tf.reduce_mean(self.temp_loss)

                self.loss.append(
                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.input_y[i])))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]

            # tf.summary.scalar('loss',self.total_loss)
            # tf.scalar_summary(['loss'],[self.total_loss])
            with tf.name_scope("accuracy"):
                self.accuracy.append(
                    tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),
                                   name="accuracy"))

        # tf.summary.scalar('loss',self.total_loss)
        tf.summary.scalar('loss', self.total_loss)
        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(self.settings.regularizer),
            weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('final_loss', self.final_loss)

    def process(self,word_batch, pos1_batch, pos2_batch, y_batch):

        feed_dict = {}
        total_shape = []
        total_num = 0
        total_word = []
        total_pos1 = []
        total_pos2 = []

        for i in range(len(word_batch)):
            total_shape.append(total_num)
            total_num += len(word_batch[i])
            for word in word_batch[i]:
                total_word.append(word)
            for pos1 in pos1_batch[i]:
                total_pos1.append(pos1)
            for pos2 in pos2_batch[i]:
                total_pos2.append(pos2)

        total_shape.append(total_num)
        total_shape = np.array(total_shape)
        total_word = np.array(total_word)
        total_pos1 = np.array(total_pos1)
        total_pos2 = np.array(total_pos2)

        feed_dict[self.total_shape] = total_shape
        feed_dict[self.input_word] = total_word
        feed_dict[self.input_pos1] = total_pos1
        feed_dict[self.input_pos2] = total_pos2
        feed_dict[self.input_y] = y_batch

        loss, accuracy, prob = self.session.run(
            [self.loss, self.accuracy, self.prob], feed_dict)

        return prob, accuracy

    def get_pos_embedding(self, line, word2id):
        en1, en2, sentence = line.strip().split()
        print("实体1: " + en1)
        print("实体2: " + en2)
        print(sentence)
        en1pos = sentence.find(en1)
        if en1pos == -1:
            en1pos = 0
        en2pos = sentence.find(en2)
        if en2pos == -1:
            en2pos = 0
        pos_embedding = []
        fixlen = self.settings.num_steps
        # Encoding test x
        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = self.pos_embed(i - en1pos)
            rel_e2 = self.pos_embed(i - en2pos)
            pos_embedding.append([word, rel_e1, rel_e2])
        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            pos_embedding[i][0] = word
        return pos_embedding

    # embedding the position
    def pos_embed(self,x):
        if x < -60:
            return 0
        if -60 <= x <= 60:
            return x + 61
        if x > 60:
            return 122

    def get_batch(self, pos_embedding, relation2id):
        pos_embedding_patch = []
        pos_embedding_patch.append([pos_embedding])
        label = [0 for i in range(len(relation2id))]
        label[0] = 1
        relation_batch = []
        relation_batch.append(label)
        pos_embedding_patch = np.array(pos_embedding_patch)
        relation_batch = np.array(relation_batch)
        word_batch = []
        pos1_batch = []
        pos2_batch = []
        for i in range(len(pos_embedding_patch)):
            word = []
            pos1 = []
            pos2 = []
            for j in pos_embedding_patch[i]:
                temp_word = []
                temp_pos1 = []
                temp_pos2 = []
                for k in j:
                    temp_word.append(k[0])
                    temp_pos1.append(k[1])
                    temp_pos2.append(k[2])
                word.append(temp_word)
                pos1.append(temp_pos1)
                pos2.append(temp_pos2)
            word_batch.append(word)
            pos1_batch.append(pos1)
            pos2_batch.append(pos2)
        word_batch = np.array(word_batch)
        pos1_batch = np.array(pos1_batch)
        pos2_batch = np.array(pos2_batch)

        batch = Batch(word_batch, pos1_batch, pos2_batch, relation_batch)

        return batch

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.settings.cell_type == RNN__CELL_TYPE.LSTM:
            cell_tmp = tf.contrib.rnn.LSTMCell(self.settings.hidden_unit)
        elif self.settings.cell_type == RNN__CELL_TYPE.GRU:
            cell_tmp = tf.contrib.rnn.GRUCell(self.settings.hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.is_training and self.settings.keep_prob < 1:
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.settings.keep_prob)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.settings.keep_prob)
        return cell_fw, cell_bw

class Batch:
    def __init__(self, word_batch, pos1_batch, pos2_batch, relation_batch):
        self.word_batch = word_batch
        self.pos1_batch = pos1_batch
        self.pos2_batch = pos2_batch
        self.relation_batch = relation_batch

class RNN__CELL_TYPE(Enum):
      LSTM = 1
      GRU = 2