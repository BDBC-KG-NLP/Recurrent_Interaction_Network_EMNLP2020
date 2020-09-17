"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
import random
from collections import Counter

random.seed(1234)
np.random.seed(1234)

VOCAB_PREFIX = ['[PAD]', '[UNK]']

def parse_args():
    # 300-dimensional glove embedding is used for nyt10 and nyt11
    parser.add_argument('--dataset', default='nyt10', help='dataset directory nyt10 or nyt11.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    train_file = 'dataset/'+args.dataset+'/train.json'
    test_file = 'dataset/'+args.dataset+'/test.json'
    wv_file = 'dataset/glove/'+args.wv_file
    wv_dim = args.wv_dim

    vocab_file = 'dataset/'+args.dataset+'/vocab.pkl'
    emb_file = 'dataset/'+args.dataset+'/embedding.npy'

    print("loading tokens...")
    train_tokens, train_pos, train_label = load_tokens(train_file)
    test_tokens, test_pos, test_label = load_tokens(test_file)
    
    print("loading glove words...")
    glove_vocab = load_glove_vocab(args.wv_file, args.wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")
    v = build_vocab(train_tokens, glove_vocab)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))

    print("building embeddings...")
    embedding = build_embedding(args.wv_file, v, args.wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)

    print('saving the dicts...')
    ret = dict()
    pos_list = VOCAB_PREFIX+list(set(train_pos+test_pos))
    pos_dict = {pos_list[i]:i for i in range(len(pos_list))}
    label_list = list(set(train_label+test_label))
    label_dict = {label_list[i]:i for i in range(len(label_list))}
    ret['pos'] = pos_dict
    ret['label'] = label_dict
    ret['bio'] = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

    print(ret['pos'])
    print(ret['label'])
    open('./dataset/'+args.dataset+'/constant.py', 'w').write(str(ret))

    print("all done.")

def load_glove_vocab(filename, wv_dim):
    vocab = set()
    with open('./dataset/glove/'+filename, encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            vocab.add(token)
    return vocab

def load_tokens(filename):
    data = read_json(filename)
    tokens = []
    pos = []
    label = []
    for d in data:
        ts = d['sentText']
        tokens += list(ts)
        pos += list(d['pos'])
        for rm in d['relationMentions']:
            label += [rm['label']]
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens, list(set(pos)), list(set(label))

def build_vocab(tokens, glove_vocab):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # sort words according to its freq
    v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens
    v = VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0 # pad vector
    w2id = {w: i for i, w in enumerate(vocab)}
    with open('./dataset/glove/'+wv_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            a_data = json.loads(line)
            data.append(a_data)
    return data

if __name__ == '__main__':
    main()


