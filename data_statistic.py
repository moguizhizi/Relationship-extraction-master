import codecs
import tensorflow as tf
from base.helper.args import get_args_parser

def statistics(file,label_map):

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

def main(_):
    args = get_args_parser()
    label_map = update_label_map(args.rel_id_file)
    statistics('./origin_data/train.txt',label_map)
    statistics('./origin_data/test.txt', label_map)

if __name__ == "__main__":
    tf.app.run()

