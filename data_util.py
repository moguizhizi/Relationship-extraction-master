import codecs
import itertools


def statistics(file, label_map):
    for key, _ in label_map.items():
        label_map[key] = 0

    total_num = 0
    input_data = codecs.open(file, 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.strip()
        word = line.split()
        label_map[word[2]] += 1
        total_num += 1
    input_data.close()

    print(file)
    print("total_num:" + str(total_num))
    for key, value in label_map.items():
        print(key + ': %6.2f%%; ' % (100. * value / total_num))


def update_label_map(file):
    temp_label_map = {}
    input_data = codecs.open(file, 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.strip()
        word = line.split()
        temp_label_map[word[0]] = 0
    input_data.close()
    return temp_label_map


def get_max_sentence_length(file):
    input_data = codecs.open(file, 'r', 'utf-8')
    max = 0
    for line in input_data.readlines():
        line = line.strip()
        content = line.split()
        if max < len(content[3]):
            max = len(content[3])
    input_data.close()
    return max


def get_min_sentence_length(file):
    input_data = codecs.open(file, 'r', 'utf-8')
    min = 10000
    for line in input_data.readlines():
        line = line.strip()
        content = line.split()
        if min > len(content[3]):
            min = len(content[3])
    input_data.close()
    return min


def get_max_entities_length(file):
    input_data = codecs.open(file, 'r', 'utf-8')
    max = 0
    for line in input_data.readlines():
        line = line.strip()
        content = line.split()
        if max < (len(content[0]) + len(content[1])):
            max = len(content[0]) + len(content[1])
    input_data.close()
    return max


def get_sen_proportion(file, section_length):
    input_data = codecs.open(file, 'r', 'utf-8')
    sum = 0
    content_list = []
    for line in input_data.readlines():
        sum = sum + 1
        line = line.strip()
        content = line.split()
        content_list.append(len(content[3]))

    for key, group in itertools.groupby(sorted(content_list), key=lambda x: x // section_length):
        print('{}-{}: {:.2f}% '.format(key * section_length, (key + 1) * section_length - 1, (100. * len(list(group)) / sum)))

def get_entities_proportion(file, entities_length):
    input_data = codecs.open(file, 'r', 'utf-8')
    sum = 0
    content_list = []
    for line in input_data.readlines():
        sum = sum + 1
        line = line.strip()
        content = line.split()
        entities = str(content[0]) + str(content[1])
        content_list.append(len(entities))

    for key, group in itertools.groupby(sorted(content_list), key=lambda x: x // entities_length):
        print('{}-{}: {:.2f}% '.format(key * entities_length, (key + 1) * entities_length - 1, (100. * len(list(group)) / sum)))
