# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/5/28 17:36
@file: common.py
@desc: 
"""


def get_relation_id(file):
    relation2id = {}
    f = open(file, 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

    return relation2id


def get_id_relation(file):
    id2relation = {}
    f = open(file, 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        id2relation[int(content[1])] = content[0]
    f.close()

    return id2relation

def get_word_id(file):
    word2id = {}
    f = open(file, encoding='utf-8')
    f.readline()
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    return word2id

def get_word_embedding(file):
    vec = []
    f = open(file, encoding='utf-8')
    f.readline()
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        content = content[1:]
        content = [(float)(i) for i in content]
        vec.append(content)
    f.close()

    return vec

def get_relation_num(file):
    num = 0
    f = open(file, 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
           break
        num = num + 1
    f.close()

    return num