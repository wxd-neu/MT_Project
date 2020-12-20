import torch
import logging
import random
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec

# define hyper-parameters
FOLD_NUM = 10
DATA_FILE_PATH = './train_set.csv'
NUM_FEATURES = 100  # the dimension of every word
NUM_WORKERS = 8     # number of thread to run parallel

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def set_seed():
    # set seed
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def all_data2fold(fold_num=10, num=100000):
    """
    split the data to N fold
    """
    set_seed()
    fold_data = []
    # read data from file and split to texts and labels
    train_df = pd.read_csv(DATA_FILE_PATH, sep='\t', encoding='UTF-8', nrows=num)
    texts = train_df['text'].tolist()
    labels = train_df['label'].tolist()

    # 随机打乱数据的分布
    total_len = len(labels)
    index = list(range(total_len))
    np.random.shuffle(index)
    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    # 把不同类别的的文本index保存在一个字典中
    label2id = {}
    for i in range(total_len):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    # 对所有的index进行十等分，保存在all_index二维数组中，每个类别十等分
    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        cur_batch_start = 0
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            cur_batch_end = cur_batch_start + cur_batch_size
            batch_data = data[cur_batch_start:cur_batch_end]
            all_index[i].extend(batch_data)
            cur_batch_start = cur_batch_end

    for fold_index in range(fold_num):
        fold_texts = [all_texts[i] for i in all_index[fold_index]]
        fold_labels = [all_labels[i] for i in all_index[fold_index]]
        # shuffle
        index = list(range(len(fold_labels)))
        np.random.shuffle(index)
        shuffle_fold_texts = [fold_texts[i] for i in index]
        shuffle_fold_labels = [fold_labels[i] for i in index]
        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))
    return fold_data


def build_train_data():
    """
    build train data for Word2Vec
    :return: train_set
    """
    fold_data = all_data2fold(FOLD_NUM)
    fold_id = 9
    train_texts = []
    for i in range(0, fold_id):
        data = fold_data[i]
        train_texts.extend(data['text'])
    logging.info('Total %d docs.' % len(train_texts))
    return train_texts


def train_word2vec(num_features=100, num_workers=8)
    train_texts = build_train_data()
    logging.info('Start training...')
    train_texts = list(map(lambda x: list(x.split()), train_texts))
    model = Word2Vec(train_texts, workers=num_workers, size=num_features, min_count=2)
    model.wv.init_sims(replace=True)

    # save word2vec
    model.save("./word2vec/word2vec_10000.bin")
    # load model
    model = Word2Vec.load("./word2vec.bin")
    # convert format
    model.wv.save_word2vec_format('./word2vec/word2vec.txt', binary=False)


if __name__ == "__main__":
    train_word2vec(NUM_FEATURES, NUM_WORKERS)











