import numpy as np
import os
import codecs
from base.helper.args import get_args_parser



# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag

def delete_repeat(filename, filetempname):

    if os.path.exists(filetempname):
        os.remove(filetempname)

    fin = open(filename, 'r', encoding='utf-8')
    fout = codecs.open(filetempname, 'w', 'utf-8')

    entity_pair = {}

    while True:
        content = fin.readline()
        is_write = True
        if content == '':
            break

        content = content.strip().split()
        en1 = content[0]
        en2 = content[1]
        sentence = content[3]
        sentence = sentence.strip()
        tup = (en1, en2)

        if tup not in entity_pair:
            entity_pair[tup] = []
            entity_pair[tup].append(sentence)
        else:
            for temp_sen in entity_pair[tup]:
                temp_sen = temp_sen.strip()
                if temp_sen == sentence:
                    is_write = False

        if is_write == True:
            record = str(en1) + " " + str(en2) + " " + str(content[2]) + " " + str(sentence)
            fout.write(record)
            fout.write('\n')

    fin.close()
    fout.close()

    os.remove(filename)
    os.rename(filetempname,filename)

# reading data
def init():

    delete_repeat('./origin_data/test.txt', './origin_data/test_temp.txt')
    delete_repeat('./origin_data/train.txt', './origin_data/train_temp.txt')

    args = get_args_parser()

    print('reading word embedding data...')
    vec = []
    word2id = {}
    f = open('./origin_data/vec.txt', encoding='utf-8')
    content = f.readline()
    content = content.strip().split()
    dim = int(content[1])
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [(float)(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    #添加UNK，BLANK对应的vector
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

    print('reading relation to id')
    relation2id = {}
    f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

    fixlen = args.max_sentence_len
    # max length of position embedding is 60 (-60~+60)
    maxlen = 60

    train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector

    print('reading train data...')
    f = open('./origin_data/train.txt', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split()
        # get entity name
        en1 = content[0]
        en2 = content[1]
        relation = 0
        if content[2] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[2]]
        # put the same entity pair sentences into a dict
        tup = (en1, en2)

        label_tag = 0
        if tup not in train_sen:
            train_sen[tup] = []
            train_sen[tup].append([])
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            train_ans[tup] = []
            train_ans[tup].append(label)
        else:
            y_id = relation
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1

            temp = find_index(label, train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                label_tag = len(train_ans[tup]) - 1
                train_sen[tup].append([])
            else:
                label_tag = temp

        sentence = content[3]
        # print("sentence:" + str(sentence))

        en1pos = 0
        en2pos = 0
        
        #For Chinese
        en1pos = sentence.find(en1)
        if en1pos == -1:
            en1pos = 0
        en2pos = sentence.find(en2)
        if en2pos == -1:
            en2post = 0

        entity_vec = get_entity_vec(en1, en2, args.max_entities_len, word2id)
        
        output = []

        #Embeding the position
        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            temp = []
            temp.append(word)
            temp.append(rel_e1)
            temp.append(rel_e2)
            temp.append(entity_vec)
            output.append(temp)

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            output[i][0] = word

        train_sen[tup][label_tag].append(output)

    print('reading test data ...')

    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)

    f = open('./origin_data/test.txt', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split()
        en1 = content[0]
        en2 = content[1]
        relation = 0
        if content[2] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[2]]
        tup = (en1, en2)

        if tup not in test_sen:
            test_sen[tup] = []
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            test_ans[tup] = label
            # print("test_ans[tup]:" + str(test_ans[tup]))
            # print("test_ans[tup]:" + str(test_ans[tup][0]))
        else:
            y_id = relation
            # print("test_ans[tup]" + str(test_ans[tup]))
            test_ans[tup][y_id] = 1

        sentence = content[3]

        en1pos = 0
        en2pos = 0
        
        #For Chinese
        en1pos = sentence.find(en1)
        if en1pos == -1:
            en1pos = 0
        en2pos = sentence.find(en2)
        if en2pos == -1:
            en2post = 0
            
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word
        test_sen[tup].append(output)

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    print('organizing train data')
    f = open('./data/train_q&a.txt', 'w', encoding='utf-8')
    temp = 0

    for i in train_sen:
        if len(train_ans[i]) != len(train_sen[i]):
            print('ERROR')
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            # print("train_y:" + str(train_y))
            f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    f.close()

    # print("train_x:" + str(train_x))

    print('organizing test data')
    f = open('./data/test_q&a.txt', 'w', encoding='utf-8')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    f.close()

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    np.save('./data/vec.npy', vec)
    np.save('./data/train_x.npy', train_x)
    np.save('./data/train_y.npy', train_y)
    np.save('./data/testall_x.npy', test_x)
    np.save('./data/testall_y.npy', test_y)

def seperate():
    print('reading training data')
    x_train = np.load('./data/train_x.npy',allow_pickle=True)

    train_word = []
    train_pos1 = []
    train_pos2 = []
    train_entities = []

    print('seprating train data')
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        entities = []
        for j in x_train[i]:
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
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)
        train_entities.append(entities)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    train_entities = np.array(train_entities)

    np.save('./data/train_word.npy', train_word)
    np.save('./data/train_pos1.npy', train_pos1)
    np.save('./data/train_pos2.npy', train_pos2)
    np.save('./data/train_entities.npy', train_entities)

    print('seperating test all data')
    x_test = np.load('./data/testall_x.npy',allow_pickle=True)
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
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
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./data/testall_word.npy', test_word)
    np.save('./data/testall_pos1.npy', test_pos1)
    np.save('./data/testall_pos2.npy', test_pos2)

# get answer metric for PR curve evaluation
def getans():
    test_y = np.load('./data/testall_y.npy',allow_pickle=True)
    # print("test_y:" + str(test_y))
    eval_y = []
    for i in test_y:
        # print("i[1:]:" + str(i[1:]))
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    # print("allans:" + str(allans))
    np.save('./data/allans.npy', allans)


def get_metadata():
    fwrite = open('./data/metadata.tsv', 'w', encoding='utf-8')
    f = open('./origin_data/vec.txt', encoding='utf-8')
    f.readline()
    while True:
        content = f.readline().strip()
        if content == '':
            break
        name = content.split()[0]
        fwrite.write(name + '\n')
    f.close()
    fwrite.close()

def get_entity_vec(entity_h, entity_t, length, word2id):
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

    return  standard_entitis


init()
seperate()
getans()
get_metadata()

print("Sucess!!!")

