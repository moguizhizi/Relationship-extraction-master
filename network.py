from enum import Enum

import numpy as np
import tensorflow as tf


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
        # 实体对的长度
        self.entities_len = 30
        # 权重类型
        self.weight_type = WEIGHT_TYPE.NORMAL
        # 句子vec类型
        self.representation_type = REPRESATATION_TYPE.VECTOR_SUM


class RNN_MODEL:
    def __init__(self, is_training, word_embeddings, settings, session):

        self.session = session
        self.settings = settings

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_steps], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_steps], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_steps], name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.settings.num_classes], name='input_y')
        self.input_entities = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.entities_len],
                                             name='input_entities')
        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[self.settings.big_num + 1], name='total_shape')
        self.total_num = self.total_shape[-1]

        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0

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

        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []

        output_h = self.get_word_feature(pos1_embedding, pos2_embedding, word_embedding)

        words_weight = self.get_words_weight(attention_w, output_h, word_embedding, self.settings.weight_type)

        attention_r = self.get_sentence_representaion(words_weight, output_h)

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

            with tf.name_scope("accuracy"):
                self.accuracy.append(
                    tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),
                                   name="accuracy"))

        tf.summary.scalar('loss', self.total_loss)
        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(self.settings.regularizer),
            weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('final_loss', self.final_loss)

    def get_words_weight(self, attention_w, output_h, word_embedding, weight_type):

        if weight_type == WEIGHT_TYPE.NORMAL:
            similar = tf.matmul(
                tf.reshape(tf.tanh(output_h), [self.total_num * self.settings.num_steps, self.settings.hidden_unit]),
                attention_w)
        elif weight_type == WEIGHT_TYPE.RELATION:
            relation_info = self.get_relation_feature(word_embedding)
            similar = tf.matmul(tf.tanh(output_h),
                                tf.reshape(relation_info, [self.total_num, self.settings.hidden_unit, 1]))

        words_weight = tf.reshape(tf.nn.softmax(
            tf.reshape(similar, [self.total_num, self.settings.num_steps])),
            [self.total_num, 1, self.settings.num_steps])

        return words_weight

    def get_word_feature(self, pos1_embedding, pos2_embedding, word_embedding):

        temp_forward = []
        temp_backward = []

        for i in range(self.settings.num_layers):
            print("i:" + str(i))
            rnn_cell_forward, rnn_cell_backward = self._bi_dir_rnn()
            temp_forward.append(rnn_cell_forward)
            temp_backward.append(rnn_cell_backward)

        cell_forward = tf.contrib.rnn.MultiRNNCell(temp_forward, state_is_tuple=True)
        cell_backward = tf.contrib.rnn.MultiRNNCell(temp_backward, state_is_tuple=True)

        initial_state_forward = cell_forward.zero_state(self.total_num, tf.float32)
        initial_state_backward = cell_backward.zero_state(self.total_num, tf.float32)
        # embedding layer
        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(word_embedding, self.input_word),
                                                   tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
                                                   tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)])

        inputs_backward = tf.concat(axis=2,
                                    values=[tf.nn.embedding_lookup(word_embedding, tf.reverse(self.input_word, [1])),
                                            tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1, [1])),
                                            tf.nn.embedding_lookup(pos2_embedding, tf.reverse(self.input_pos2, [1]))])

        outputs_forward = []
        state_forward = initial_state_forward
        with tf.variable_scope('RNN_FORWARD') as scope:
            for step in range(self.settings.num_steps):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_forward, state_forward) = cell_forward.call(inputs_forward[:, step, :], state_forward)
                outputs_forward.append(cell_output_forward)

        outputs_backward = []
        state_backward = initial_state_backward
        with tf.variable_scope('RNN_BACKWARD') as scope:
            for step in range(self.settings.num_steps):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_backward, state_backward) = cell_backward.call(inputs_backward[:, step, :], state_backward)
                outputs_backward.append(cell_output_backward)

        output_forward = tf.reshape(tf.concat(axis=1, values=outputs_forward),
                                    [self.total_num, self.settings.num_steps, self.settings.hidden_unit])
        output_backward = tf.reverse(tf.reshape(tf.concat(axis=1, values=outputs_backward),
                                                [self.total_num, self.settings.num_steps, self.settings.hidden_unit]),
                                     [1])
        # word-level attention layer
        output_h = tf.add(output_forward, output_backward)

        return output_h

    def process(self, word_batch, pos1_batch, pos2_batch, y_batch, entities_batch):

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
            for entities in entities_batch[i]:
                total_entities.append(entities)

        total_shape.append(total_num)
        total_shape = np.array(total_shape)
        total_word = np.array(total_word)
        total_pos1 = np.array(total_pos1)
        total_pos2 = np.array(total_pos2)
        total_entities = np.array(total_entities)

        feed_dict[self.total_shape] = total_shape
        feed_dict[self.input_word] = total_word
        feed_dict[self.input_pos1] = total_pos1
        feed_dict[self.input_pos2] = total_pos2
        feed_dict[self.input_y] = y_batch
        feed_dict[self.input_entities] = total_entities

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

        entity_vec = self.get_entity_vec(en1, en2, self.settings.entities_len, word2id)

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = self.pos_embed(i - en1pos)
            rel_e2 = self.pos_embed(i - en2pos)
            temp = []
            temp.append(word)
            temp.append(rel_e1)
            temp.append(rel_e2)
            temp.append(entity_vec)
            pos_embedding.append(temp)

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            pos_embedding[i][0] = word
        return pos_embedding

    # embedding the position
    def pos_embed(self, x):
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
        entities_batch = []
        for i in range(len(pos_embedding_patch)):
            word = []
            pos1 = []
            pos2 = []
            entities = []
            for j in pos_embedding_patch[i]:
                temp_word = []
                temp_pos1 = []
                temp_pos2 = []
                temp_entities = []
                for k in j:
                    temp_word.append(k[0])
                    temp_pos1.append(k[1])
                    temp_pos2.append(k[2])
                    temp_entities = k[3]
                word.append(temp_word)
                pos1.append(temp_pos1)
                pos2.append(temp_pos2)
                entities.append(temp_entities)
            word_batch.append(word)
            pos1_batch.append(pos1)
            pos2_batch.append(pos2)
            entities_batch.append(entities)
        word_batch = np.array(word_batch)
        pos1_batch = np.array(pos1_batch)
        pos2_batch = np.array(pos2_batch)
        entities_batch = np.array(entities_batch)

        batch = Batch(word_batch, pos1_batch, pos2_batch, relation_batch, entities_batch)

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

    def _dir_rnn(self):
        """
        单向RNN
        :return:
        """
        cell_single_fw = self._witch_cell()
        if self.is_training and self.settings.keep_prob < 1:
            cell_single_fw = tf.contrib.rnn.DropoutWrapper(cell_single_fw, output_keep_prob=self.settings.keep_prob)
        return cell_single_fw

    def get_relation_feature(self, word_embedding):

        temp_fw = []
        for i in range(self.settings.num_layers):
            rnn_single_fw = self._dir_rnn()
            temp_fw.append(rnn_single_fw)

        cell_forward = tf.contrib.rnn.MultiRNNCell(temp_fw)
        initial_state_forward = cell_forward.zero_state(self.total_num, tf.float32)

        inputs_forward = tf.nn.embedding_lookup(word_embedding, self.input_entities)

        state_forward = initial_state_forward
        with tf.variable_scope('RELATION_RNN_FORWARD') as scope:
            for step in range(self.settings.entities_len):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_forward, state_forward) = cell_forward.call(inputs_forward[:, step, :], state_forward)

        return cell_output_forward

    def get_entity_vec(self, entity_h, entity_t, length, word2id):
        entitis = str(entity_h) + " " + str(entity_t)
        standard_entitis = []
        for i in range(length):
            word = word2id['BLANK']
            standard_entitis.append(word)

        for i in range(min(length, len(entitis))):
            word = 0
            if entitis[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[entitis[i]]

            standard_entitis[i] = word

        return standard_entitis

    def get_sentence_representaion(self, words_weight, output_h):

        if self.settings.representation_type == REPRESATATION_TYPE.VECTOR_SUM:
            attention_r = tf.reshape(tf.matmul(words_weight, output_h),
                                     [self.total_num, self.settings.hidden_unit])
        elif self.settings.representation_type == REPRESATATION_TYPE.MAX_POOLING:
            temp = tf.transpose(output_h, perm=[0, 2, 1])
            attention_r = tf.reduce_max(tf.multiply(words_weight, temp), 2)

        return attention_r


class Batch:
    def __init__(self, word_batch, pos1_batch, pos2_batch, relation_batch, entities_batch):
        self.word_batch = word_batch
        self.pos1_batch = pos1_batch
        self.pos2_batch = pos2_batch
        self.relation_batch = relation_batch
        self.entities_batch = entities_batch


class RNN__CELL_TYPE(Enum):
    LSTM = 1
    GRU = 2


class WEIGHT_TYPE(Enum):
    NORMAL = 1
    RELATION = 2


class REPRESATATION_TYPE(Enum):
    VECTOR_SUM = 1
    MAX_POOLING = 2
