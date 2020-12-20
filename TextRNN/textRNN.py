import time
import random
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from transformers import BasicTokenizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# basic_tokenizer = BasicTokenizer()


def set_hyperparameters():
    """
    为模型设置超参数，把所有的超参数添加到一个dict中，之后需要使用超参数就可以
    直接从这个超参数的字典中获取，对超参数进行统一的管理，提高了代码的整洁性和
    可读性
    :return: hyperparameters dict
    """
    hyper_params = {}

    # 把超参数添加到dict中
    hyper_params['DATA_FILE_PATH'] = '../data/train_set.csv'
    hyper_params['TEST_DATA_FILE_PATH'] = '../data/test_a.csv'
    hyper_params['FOLD_NUM'] = 10  # n折交叉验证
    hyper_params['NUM'] = 1000  # 需要的训练集大小
    hyper_params['WORD2VEC_PATH'] = '../word2vec/word2vec.txt'
    hyper_params['CNNENCODER_DROPOUT'] = 0.15
    hyper_params['LR'] = 2e-4
    hyper_params['DECAY'] = 0.75
    hyper_params['DECAY_STEP'] = 1000
    hyper_params['SAVE_MODEL_PATH'] = './save_file/cnn.bin'
    hyper_params['SAVE_TEST_PATH'] = './save_file/cnn.csv'

    return hyper_params


def set_seed():
    """
    set seed for entire project
    :return: none
    """
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_cuda(gpu_id=0):
    use_cuda = gpu_id >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device("cpu")
    print("Use cuda: %s, gpu id: %d" % (use_cuda, gpu_id))
    return use_cuda, device


class DataProcessLoader():
    """
    这个类是对于原始数据的一系列处理，使得得到最后模型训练所需要的输入格式化样本
    对于每个函数的功能看每个函数的注释部分
    """
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.fold_num = hyper_params['FOLD_NUM']
        self.fold_data = self.all_data2fold(hyper_params['DATA_FILE_PATH'], fold_num=self.fold_num)
        self.train_data, self.dev_data, self.test_data, vocab_data = self.build_data(self.fold_data, self.fold_num - 1)
        self.vocab = Vocab(vocab_data)
        self.extword_embed = self.vocab.load_pretrained_embs(hyper_params['WORD2VEC_PATH'])

        self.train_examples = self.get_examples(self.train_data, self.vocab)
        self.dev_examples = self.get_examples(self.dev_data, self.vocab)
        # self.test_examples = self.get_examples(self.test_data, self.vocab)

    def all_data2fold(self, data_file_path, fold_num=10, data_num=10000):
        """
        split the data to N fold
        把原始的数据集平分为fold_num份，方便后面做交叉验证
        :return: fold_data
        返回fold_data是一个list，包含十个元素，每一个元素是一个size为data_num/fold_num的字典
        这里的data_num是指新闻文本的个数，fold_num是指对于全部数据分为fold_num份
        字典中包含两个k-v，key是'label'和'text'，value都是一个list，这两个list是一一对应
        的关系，其中text的value列表每一个值为一个新闻文本，label的value列表每一个值为一个新闻
        对应的label值
        """
        fold_data = []
        # read data from file and split to texts and labels
        train_df = pd.read_csv(data_file_path, sep='\t', encoding='UTF-8', nrows=data_num)
        texts = train_df['text'].tolist()
        labels = train_df['label'].tolist()

        # 随机打乱数据的分布，在合成一份完整数据的list
        total_len = len(labels)
        data_index = list(range(total_len))
        np.random.shuffle(data_index)
        all_texts = []
        all_labels = []
        for i in data_index:
            all_texts.append(texts[i])
            all_labels.append(labels[i])

        # 遍历all_labels把不同的label的index添加到对应的value_list中
        # 作用是为后续对于把每个label的数据fold_num等分做准备
        data_index_labels = {}
        for i in range(total_len):
            label = str(all_labels[i])
            if label not in data_index_labels:
                data_index_labels[label] = [i]
            else:
                data_index_labels[label].append(i)

        # 对所有的index进行fold_num等分，保存在all_index二维数组中
        # 每个类别fold_num等分,在最后合成所有数据的fold_num等分
        all_index = [[] for _ in range(fold_num)]
        for label, label_data_index in data_index_labels.items():
            batch_size = len(label_data_index) // fold_num
            other = len(label_data_index) - batch_size * fold_num
            cur_batch_start = 0
            for i in range(fold_num):
                cur_batch_size = batch_size + 1 if i < other else batch_size
                cur_batch_end = cur_batch_start + cur_batch_size
                batch_data = label_data_index[cur_batch_start:cur_batch_end]
                all_index[i].extend(batch_data)
                cur_batch_start = cur_batch_end

        for fold_index in range(fold_num):
            fold_texts = [all_texts[i] for i in all_index[fold_index]]
            fold_labels = [all_labels[i] for i in all_index[fold_index]]
            # 对于每份数据进行打乱
            fold_index = list(range(len(fold_labels)))
            np.random.shuffle(fold_index)
            shuffle_fold_texts = [fold_texts[i] for i in fold_index]
            shuffle_fold_labels = [fold_labels[i] for i in fold_index]
            data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
            fold_data.append(data)

        print("Fold lens %s" % str([len(data['label']) for data in fold_data]))

        return fold_data

    def build_data(self, fold_data, dev_fold_index):
        """
        build train, dev, test data
        the shape of dev 、train 、test is same, but sizes of them are different
        the type of them is dict, {'text': value_list, 'label': value_list}
        the type of text value list elemet is str
        the baseline just use the train_data to build vocab but if we use all trian_data
        dev_data and test_data to build a vocab it may works better.

        :fold_data: 参考all_data2fold的返回值
        :fold_index: 取出索引为fold_data[fold_index]为dev data
        :return: train_data dev_data test_data
        """

        # dev data
        dev_data = fold_data[dev_fold_index]

        # train data
        train_texts = []
        train_labels = []

        # 获得训练数据对应的fold_index
        train_fold_index = [i for i in range(0, dev_fold_index)]
        train_fold_index += [i for i in  range(dev_fold_index + 1, self.fold_num)]

        # 把需要的训练数据从fold_data列表中取出并且合并
        for index in train_fold_index:
            train_texts.extend(fold_data[index]['text'])
            train_labels.extend(fold_data[index]['label'])
        train_data = {'text': train_texts, 'label': train_labels}

        # test data
        test_df = pd.read_csv(self.hyper_params['TEST_DATA_FILE_PATH'], sep='\t', encoding='UTF-8')
        texts = test_df['text'].tolist()
        test_data = {'text': texts, 'label': [0] *  len(texts)}

        # vocab data
        vocab_data = {'text': [], 'label': []}
        vocab_data['text'].extend(train_data['text'])
        vocab_data['label'].extend(train_data['label'])
        vocab_data['text'].extend(dev_data['text'])
        vocab_data['label'].extend(dev_data['label'])
        vocab_data['text'].extend(test_data['text'])
        vocab_data['label'].extend(test_data['label'])

        return train_data, dev_data, test_data, vocab_data

    def sentence_split(self, text, vocab, max_sent_len=256, max_segment=16):
        """
        输入的text表示一篇新闻，最后返回的segments是一个list，其中每个元素是一个tuple：(句子长度，句子本身)。
        根据一篇文章，把文章分为多个句子，
        :param text: 一篇新闻
        :param vocab: 词典
        :param max_sent_len: 每句话的最大长度
        :param max_segment: 一篇新闻最多包含的句子数
        :return: segments [sent_num * (sent_len, sent<list>)]
        """

        words = text.strip().split()
        document_len = len(words)

        # 每个句子第一个word的index，保存在一个list中
        index = list(range(0, document_len, max_sent_len))
        index.append(document_len)

        segments = []
        for i in range(len(index) - 1):
            sent_len = index[i + 1] - index[i]
            # 根据索引划分句子
            segment = words[index[i]: index[i + 1]]
            # 如果是低频word替换为UNK
            segment = [word if word in vocab._id2word else 'UNK' for word in segment]
            segments.append((sent_len, segment))

        if len(segments) > max_segment:
            half_segment= int(max_segment / 2)
            return segments[:half_segment] + segments[-half_segment:]
        else:
            return segments

    def get_examples(self, data, vocab, max_sent_len=256, max_segment=8):
        """
        遍历每一篇新闻，对每一篇新闻调用sentence_split进行句子划分
        :param data: 需要划分的全部新闻数据
        :param vocab: 词表
        :param max_sent_len: 最大的句子长度
        :param max_segment: 每篇新闻划分的最大句子数目
        :return: 一个list，每个元素是一个tuple(label, doc_len, doc)
                其中doc_len是句子的数量，doc又是一个list,每个元素是一个tuple
                (sent_len, word_ids, extword_ids)
                examples : [doc_nums * (label_id, doc_len, doc[sent_num *
                           (sent_len, word_ids<list>, extword_ids<list>)])]
        """

        label2id = vocab.label2id
        examples = []
        for text, label in zip(data['text'], data['label']):
            id = label2id(label)
            # segments是一个list，其中每个元素是一个tuple(sent_len, sent_word)
            segments = self.sentence_split(text, vocab, max_sent_len, max_segment)
            doc = []
            for sent_len, sent_words in segments:
                # 把words转化为ids
                word_ids = vocab.word2id(sent_words)
                # 把words转化为ext——ids
                extword_ids = vocab.extword2id(sent_words)
                doc.append((sent_len, word_ids, extword_ids))
            examples.append((id, len(doc), doc))

        return examples

    @classmethod
    def batch_slice(cls, data, batch_size):
        """
        build data loader
        把数据分割为多个batch，组成一个list返回
        :param data: 是get_examles()得到的examples
        :param batch_size: 批次大小
        :return: [batch_size * (label_id, doc_len, doc[sent_num *
                           (sent_len, word_ids<list>, extword_ids<list>)])]
        """

        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for i in range(batch_num):
            if i < (batch_num - 1):
                batch_docs = data[(i * batch_size):((i + 1) * batch_size)]
            else:
                batch_docs = data[(i * batch_size):]

            # 和return类似，也可以返回数据，但是是分批次的返回，可以节省内存
            # 用next)可以接着上次继续执行再返回数据
            yield batch_docs

    @classmethod
    def data_iter(cls, data, batch_size, shuffle=True, noise=1.0):
        """
        在迭代训练中，调用data_iter函数，生成每个批次的batch_data，这个函数
        中会调用batch_slice来获取batch_data的原始数据

        :param data: get_examples()得到的结果，格式见get_example的返回
        :param batch_size: 批次大小
        :param shuffle: 是否为乱序
        :param noize:
        :return:
        """
        batched_data = []
        if shuffle:
            # 打乱所有数据
            np.random.shuffle(data)
            # lengths表示每篇文章的句子数量
            lengths = [example[1] for example in data]
            # 不知道这步有什么实质性的作用？
            noisy_lengths = [- (l + np.random.uniform(- noise, noise))
                             for l in lengths]
            sorted_indices = np.argsort(noisy_lengths).tolist()
            sorted_data = [data[i] for i in sorted_indices]
        else:
            sorted_data = data

        # 把batch的数据放在一个list中
        batched_data.extend(list(DataProcessLoader.batch_slice(sorted_data, batch_size)))

        if shuffle:
            np.random.shuffle(batched_data)

        for batch_data in batched_data:
            yield batch_data


class Vocab():
    """
    Vocab 的作用是：
    1. 创建word和index对应的字典，这里包括2份字典，分别是：_id2word 和 _id2extword
    其中_id2word是从新闻得到的，把词频小于5的词替换为了UNK, 对应到模型输入的batch_inputs1。
    _id2extword是从word2vec.txt中得到的, 对应到模型输入的batch_inputs2。后面会有两个
    embedding层，其中_id2word对应的embedding是可学习的，_id2extword对应的embedding是从
    文件中加载的，是固定的
    2. 创建label和index对应的字典
    """
    def __init__(self, train_data):
        self.min_count = 5
        self.pad = 0  # 填充的word对应的index
        self.unk = 1  # 未出现在词典中的word对应的index
        self._id2word = ['PAD', 'UNK']
        self._id2extword = ['PAD', 'UNK']
        self._id2label = []
        self.target_names = []
        self.word_counter = Counter()

        self.build_vocab(train_data)
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)

        print("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    def build_vocab(self, train_data):
        """
        用train_data创建词典
        :param train_data: build_data()的到的train_data
        :return: None
        """
        for text in train_data['text']:
            words = text.split()     # return a list of str
            tmp_cnt = Counter(words)
            self.word_counter += tmp_cnt

        # 对于train data中的所有word用Counter()进行统计，按照count从大到小进行排序
        # 后，依次加入self._id2word中(是一个list), 用index指示每个word的id, 这里的
        # word类型是str，就是word对应的数字编号str
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        self.label_counter = Counter(train_data['label'])
        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])
        # print('self._id2label:', self._id2label)
        # print('self.target_names', self.target_names)

    def load_pretrained_embs(self, embfile_path):
        """
        加载预训练文件，并做一定的处理
        通过index在self._id2extword中可以找到word
        同时，通过index可以在embeddings中找到词向量
        同一个index找到的word和词向量是对应的
        对于不在词表中的word采用所有出现的词向量的平均值代替
        :param embfile_path: 预训练好的词向量文件路径
        :return: embeddings
        """

        # 打开Word2vec的.txt文件，文件中第一行 'word_num embedding_dim'
        # 然后接下来的每一行为 'word word2vec', 详细看word2vec.txt文件
        with open(embfile_path, encoding='utf-8') as file:
            lines = file.readlines()  # read all lines and return a list
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])
        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim), dtype=np.float32)
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector   # 对于np.array可以把每个词向量相加
            embeddings[index] = vector
            index += 1

        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)  # 这是干嘛的？

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        embeddings = torch.from_numpy(embeddings)

        return embeddings

    # 通过word找到对应的index，对于没有出现的index用UNK表示
    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    # 同上
    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    # 同上
    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)


class WordLSTMEncoder(nn.Module):
    """
    使用一个双层的LSTM对于单词进行编码，这里采取的是BiLSTM + Attention 对句子进行编码
    """

    def __init__(self, vocab, extword_embeddings, dropout=0.15, word_hidden_size=128):
        super(WordLSTMEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = 100

        self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)
        self.extword_embed = nn.Embedding.from_pretrained(extword_embeddings, freeze=True,
                                                          padding_idx=0)

        self.word_bilstm = nn.LSTM(input_size=self.word_dims,
                                   hidden_size=word_hidden_size,
                                   num_layers=2,
                                   bias=True,
                                   batch_first=True,
                                   bidirectional=True)

    def forward(self, word_ids, extword_ids, batch_masks):
        """
        输入一个批次的数据，其中包括对应的word_embed中的indexs和extword_embed的indexs，
        batch_masks指出当前的这个word是否为padding word，如果是则相应位置为0，不是则为1

        :param word_ids: (batch_size * sent_nums(max_doc_len), max_sent_len)
        :param extword_ids: 同上
        :param batch_masks: 同上
        :return: sent_reps    (batch_size * sent_nums, sent_reps_size)
                              其中的sent_reps其实就是biLstm的hn拼接，维度为2 * lstm_hidden_size
        """

        word_embed = self.word_embed(word_ids)
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)

        word_hiddens, _ = self.word_bilstm(batch_embed)
        word_hiddens = word_hiddens * batch_masks.unsqueeze(2)

        if self.training:
            word_hiddens = self.dropout(word_hiddens)

        return word_hiddens


class SentEncoder(nn.Module):
    """
    用2层的BiLSTM对WordLSTMEncoder + Attention层编码后的句子向量进行处理，隐藏层大小默认为256，目的是
    最终得到一个新闻doc的向量表示，但是由于加了attention机制，所以这个网络的输出为bilstm的output而非最后
    的hn，常规的做法是把最后一层的bilstm的hn进行拼接得到整个句子的向量表示，但是会造成新闻中部分信息的丢失，
    预测效果降低，所以这里为了后续的attention层处理方便，返回的是整个output。
    """
    def __init__(self, sent_rep_size=256, dropout=0.15, sent_hidden_size=256, num_layers=2):
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # sent_rep_size = 256就是WordLSTMEncoder的隐藏层大小的2倍
        self.sent_bilstm = nn.LSTM(input_size=sent_rep_size,
                                   hidden_size=sent_hidden_size,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   bidirectional=True)

    def forward(self, sent_reps, sent_masks):
        """
        对于详细的数据流向和维度解释，看我的笔记参考博客，以及官方文档的解释
        :return: 最后的一层的输出
        """

        # sent_reps: batch_size * doc_len * sent_rep_size
        # sent_masks: batch_size * doc_len  把没有单词的句子设置为0表示
        # sent_hiddens: batch, seq_len, num_directions * hidden_size
        sent_hiddens, _ = self.sent_bilstm(sent_reps)
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens


class Attention(nn.Module):
    """
    Attention的输入是sent_hiddens和sent_masks，首先sent_hiddens经过线性变换得到key值，维度不变，
    所以key的shape是：(batch_size, doc_len, num_directions * hidden_size)，然后key和query相乘
    得到最后的outputs，query的维度在这里是512，所以outputs的shape为(batch_size, doc_len, )
    """
    def __init__(self, bihidden_size):
        super(Attention, self).__init__()

        self.query = nn.Parameter(torch.Tensor(bihidden_size))
        nn.init.normal_(self.query, mean=0.0, std=0.05)

        self.linear = nn.Linear(bihidden_size, bihidden_size, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.linear.bias)

    def forward(self, batch_hidden, batch_masks):

        # batch_hidden: (batch_size, len, bihidden_size)
        # batch_masks: (batch_size, len)

        # linear transformation to get key
        # key: batch_size * len * hidden_size
        key = self.linear(batch_hidden)

        # compute attention value
        # matmul 会进行广播，其实是把最后一个维度自己加权求和，如果是对句子attention，
        # 则是512维度的句子向量加权求和，得到一个数表示这个句子和最后doc_vec的相关性。
        # attn_value: batch_size * len
        attn_value = torch.matmul(key, self.query)

        # batch_masks: batch_size * len
        # 1 - batch_masks 就是取反，把没有单词的句子置为 0
        # masked_fill 的作用是在为1的地方替换为value: float(-1e32)
        mask_attn_value = attn_value.masked_fill((1 - batch_masks).bool(), float(-1e32))

        # attn_weights: batch_size * doc_len
        attn_weights = F.softmax(mask_attn_value, dim=1)

        # 其实这步就是把最后的填充句子的注意力权重置为0。其实在这里可以不做这个处理，因为经过之前的
        # mask_weights的处理之后，填充句子的部分注意力权重已经很小，接近于0了
        masked_attn_weights = attn_weights.masked_fill((1 - batch_masks).bool(), 0.0)

        # 为什么这里是对attn_middle进行求和而不是对原始的batch_hidden加权求和？
        # masked_attn_weights.unsqueeze(1): batch_size * 1 * doc_len
        # attn_middle: batch_size * doc_len * hidden(512)
        # batch_outputs: batch_size * hidden(512)
        reps = torch.bmm(masked_attn_weights.unsqueeze(1), batch_hidden).squeeze(1)

        return reps, attn_weights


class Model(nn.Module):
    """
    把WordLSTMEncoder、SentEncoder、Attention、FC拼接起来搭建整体网络层
    """

    def __init__(self, vocab, use_cuda, device, extword_embed):
        super(Model, self).__init__()
        self.word_hidden_size = 128
        self.sent_reps_size =256
        self.sent_hidden_size = 256
        self.doc_reps_size = 512
        self.all_parameters = {}
        self.dropout = 0.15

        parameters = []
        self.word_encoder = WordLSTMEncoder(vocab, extword_embed, self.dropout, self.word_hidden_size)
        self.word_attention = Attention(bihidden_size=self.sent_reps_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_attention.parameters())))

        self.sent_encoder = SentEncoder(self.sent_reps_size, self.dropout, self.sent_hidden_size)
        self.sent_attention = Attention(self.doc_reps_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))

        # 最后加一层线性网络，利用doc_reps进行分类
        self.out = nn.Linear(self.doc_reps_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        print('Build model with cnn word encoder, bilstm sent encoder.')
        # self.parameter() 为一个tensor 的list，np.prob作用是把tensor的每个维度大小相乘
        # 最终得到每个tensor的大小，再通过sum相加，得到参数的个数
        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        print('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        """
        :param batch_inputs: (batch_inputs1, batch_inputs2, batch_masks)
        :return: batch_outputs：batch_size * num_labels
                其实就是对于每个新闻文本预测出对于14个标签的概率分布
        """

        # batch_inputs : batch_size * doc_len * sent_len
        # batch_masks : batch_size * doc_len * sent_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = (batch_inputs1.shape[0],
                                                 batch_inputs1.shape[1],
                                                 batch_inputs1.shape[2])
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)

        # sent_reps: (sentence_num , sentence_rep_size)
        # (sen_num, <2 * lstm_hidden_size>) =  (sen_num , 256)
        word_hiddens = self.word_encoder(batch_inputs1, batch_inputs2, batch_masks)
        sent_reps, word_atten_scores = self.word_attention(word_hiddens, batch_masks)

        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_reps_size)
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)

        # sent_masks: batch_size * doc_len
        # any(2)表示在第二个维度上判断
        # 也就是说在max_sent_len这个维度上判断，这个维度是sent的id，如果所有的
        # id都是0.说明这个句子是填充句子为了使得每篇新闻的句子总数统一到max_doc_len
        # 当为填充句子时就把这个位置置为0.0
        sent_masks = batch_masks.bool().any(2).float()

        # sent_hiddens: batch_size * doc_len * num_directions * hidden_size
        # sent_hiddens: batch, seq_len, 2 * hhidden_size
        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)

        # doc_reps: batch_size * (2 * hidden_size)
        # atten_score: batch_size * doc_len
        doc_reps, sent_atten_scores = self.sent_attention(sent_hiddens, sent_masks)

        # batch_size * num_labels
        batch_outputs = self.out(doc_reps)

        return batch_outputs


class Optimizer:
    """
    定义优化器类，对参数进行优化
    """
    def __init__(self, model_parameters, hyper_params):
        super(Optimizer, self).__init__()
        self.all_params = []
        self.optims = []
        self.schedulers = []
        self.lr = hyper_params['LR']
        self.decay = hyper_params['DECAY']
        self.decay_step = hyper_params['DECAY_STEP']

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                optim = torch.optim.Adam(parameters, lr=self.lr)
                self.optims.append(optim)

                l = lambda step: self.decay ** (step // self.decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim,
                                                              lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)

            else:
                Exception("no nameed parameters.")
        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = '  %.5f' * self.num
        res = lr % lrs
        return res


class Trainer():
    """
    定义训练器类，这个类中包括对于模型的训练，和用验证集验证，测试集进行测试，得到需要的数据
    对于准确度的测试这里采用f1-score，因为进过之前的数据分析阶段发现数据是分布不均衡的
    """
    def __init__(self, hyper_params, model=None, vocab=None):
        # 为全局设置seed和gpu
        print("train init starting")
        set_seed()
        self.use_cuda, self.device = set_cuda(0)
        dataloader = DataProcessLoader(hyper_params)

        self.vocab = vocab
        self.model = model
        if vocab is None:
            self.vocab = dataloader.vocab
        if model is None:
            self.model = Model(self.vocab, self.use_cuda, self.device, dataloader.extword_embed)

        self.report = True
        self.save_model = hyper_params['SAVE_MODEL_PATH']
        self.save_test = hyper_params['SAVE_TEST_PATH']

        self.train_batch_size = 128
        self.test_batch_size = 128
        self.train_data = dataloader.train_examples
        self.dev_data = dataloader.dev_examples
        # self.test_data= dataloader.test_examples
        # 把数据分成batch进行训练，每个批次大小为batch_size，这里的batch_num为一个有多少个批次
        self.batch_num = int(np.ceil(len(self.train_data) /
                                     float(self.train_batch_size)))

        # count
        self.epochs = 5
        self.early_stops = 3
        self.log_interval = 50
        self.clip = 5.0
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.last_epoch = self.epochs

        # define criterion
        self.criterion = nn.CrossEntropyLoss()

        # label name
        self.target_names = self.vocab.target_names

        # optimizer
        self.optimizer = Optimizer(self.model.all_parameters, hyper_params)

        print("train init finish")

    def train(self):
        print('Start trainning...')
        for epoch in range(1, self.epochs + 1):
            train_f1 = self._train(epoch)
            dev_f1 = self._eval(epoch)

            if self.best_train_f1 <= dev_f1:
                print("Exceed history dev = %.2f, current dev = %.2f" %
                      (self.best_dev_f1, dev_f1))
                torch.save(self.model.state_dict(), self.save_model)

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == self.early_stops:
                    print("Early stop in epoch %d, best train: %.2f, best dev: %.2f" %
                          (epoch, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        self.model.load_state_dict(torch.load(self.save_model))
        self._eval(self.last_epoch + 1, test=True)

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()   # see source code 其实就是把model以及子模型的trainning = True
        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []

        for batch_data in DataProcessLoader.data_iter(self.train_data, self.train_batch_size,
                                                      shuffle=True):
            torch.cuda.empty_cache()
            # batch_inputs: (batch_inputs1, batch_inputs, batch_masks)
            # shape : batch_size * doc_len * sent_len
            # batch_labels shape : batch_size
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            # batch_outputs: batch_size * num_labels(14)
            batch_outputs = self.model(batch_inputs)
            loss = self.criterion(batch_outputs, batch_labels)
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            # 把预测值转换为一维，方便之后计算f1-score
            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            # 为了防止梯度爆炸，这里采用梯度裁剪
            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=self.clip)
            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()

            self.optimizer.zero_grad()
            self.step += 1

            if batch_idx % self.log_interval == 0:
                elapsed = time.time() - start_time
                lrs = self.optimizer.get_lr()
                print(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / self.log_interval,
                        elapsed / self.log_interval))

                losses = 0
                start_time = time.time()

            batch_idx += 1

        overall_losses = overall_losses / self.batch_num
        during_time = time.time() - epoch_start_time

        # 保留4位小数
        overall_losses = float(format(overall_losses, '.4f'))
        f1 = self.get_score(y_pred, y_true)

        print(
            '| epoch {:3d} | f1 {} | loss {:.4f} | time {:.2f}'.format(epoch, f1, overall_losses, during_time))

        # 如果预测值和真实值的标签都包含相同的类别数目，才能调用classification_report
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            print('\n' + report)

        return f1

    def _eval(self, epoch, test=False):
        """
        验证集和测试集都使用这个函数来进行验证预测，用test这个参数进行区分
        :param epoch: 轮次
        :param test: True表示是测试集数据，False表示是验证集数据
        :return:
        """
        self.model.eval()   # see source code 其实就是把model以及子模型的trainning = False
        start_time = time.time()
        data = self.test_data if test else self.dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in DataProcessLoader.data_iter(data, self.test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                batch_outputs = self.model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            f1 = self.get_score(y_pred, y_true)
            during_time = time.time() - start_time

            if test:
                df = pd.DataFrame({'label': y_pred})
                df.to_csv(self.save_test, index=False, sep=',')
            else:
                print('| epoch {:3d} | dev | f1 {} | time {:.2f}'.format(epoch, f1, during_time))
                if set(y_true) == set(y_pred) and self.report:
                    report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                    print('\n' + report)

        return f1

    def batch2tensor(self, batch_data):
        """
        这步实际上是把batch_data中的数据转化为符合训练的数据格式，因为在这里传入的batch_data实际上
        里面对于每个word是以word_ids的形式给出的，其中包含了长度等信息，并不满足训练需要的格式，所以
        这里进行了一些调整，并且把数据转化为cuda的格式，方便在gpu上加速
        :param batch_data: [batch_size * (label, doc_len, doc[doc_len *
        (sent_len, word_ids, extword_ids)])]
        :return:
        """
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            doc_labels.append(doc_data[0])
            doc_lens.append(doc_data[1])
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            # 取出这篇新闻中最长的句子长度，添加到列表中
            doc_max_sent_len.append(max(sent_lens))

        # 取出这批新闻中包含句子数最大的
        max_doc_len = max(doc_lens)
        # 取出这批新闻中最长句子的句子长度
        max_sent_len = max(doc_max_sent_len)

        # 创建用于训练的数据的格式
        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(doc_labels)

        # 把对应的ids加入到batch_inputs1、batch_inputs2、batch_masks
        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                sent_data = batch_data[b][2][sent_idx]
                for word_idx in range(sent_data[0]):
                    # sent_data[1]是word_ids, sent_data[2]是extword_ids
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                    batch_masks[b, sent_idx, word_idx] = 1

        if self.use_cuda:
            batch_inputs1 = batch_inputs1.to(self.device)
            batch_inputs2 = batch_inputs2.to(self.device)
            batch_masks = batch_masks.to(self.device)
            batch_labels = batch_labels.to(self.device)

        return (batch_inputs1, batch_inputs2, batch_masks), batch_labels

    def get_score(self, y_pred, y_true):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        return f1


if __name__ == "__main__":
    hyper_params = set_hyperparameters()
    trainer = Trainer(hyper_params)
    trainer.train()






