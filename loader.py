import random
import torch
import numpy as np
from utils.helper import read_json

class DataLoader(object):
    def __init__(self, filename, batch_size, args, dicts, evaluation=False):
        self.batch_size = batch_size
        self.args = args
        self.dicts = dicts
        self.eval = evaluation # word dropout

        data = read_json(filename)

        # preprocess data
        data = self.preprocess(data, dicts, args)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))
     
    def preprocess(self, data, dicts, args):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            # tokens
            tokens = list(d['sentText'])
            if args.lower:
                tokens = [t.lower() for t in tokens]

            # pos
            pos = d['pos']

            # map to ids
            tokens = map_to_ids(tokens, dicts['token'])
            pos = map_to_ids(pos, dicts['pos'])
            l = len(tokens)    # real length of sentence

            # BIO labels
            NER_labels = [dicts['bio']['O'] for _ in range(len(tokens))]
            for en in d['en_list']:
                if len(en) == 1:
                    en_index = d['sentText'].index(en[0])
                    NER_labels[en_index] = dicts['bio']['S']
                else:
                    sta, end = find_index(d['sentText'], en)
                    tmp_block = [dicts['bio']['B']] + [dicts['bio']['I'] for _ in range(end-sta-1)] + [dicts['bio']['E']]
                    NER_labels[sta:end+1] = tmp_block

            # mask
            mask_s = [1 for i in range(l)]
            processed += [(tokens, pos, mask_s, (d['sentText'], d['relationMentions']), NER_labels)]
        return processed

    def __len__(self):
        return len(self.data)

    # 0: tokens, 1: pos, 2: mask_s, 3: RC_labels 4: NER_labels
    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 5

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, _ = sort_all(batch, lens)  

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.args.word_dropout) for sent in batch[0]]
        else:
            words = batch[0]

        # to tensors
        words = get_long_tensor(words, batch_size)
        pos = get_long_tensor(batch[1], batch_size)
        mask_s = get_float_tensor(batch[2], batch_size)
        RC_labels = [gen_labels(sentText, relationMentions, self.dicts) for sentText, relationMentions in batch[3]]
        RC_labels = padding_labels(RC_labels,  batch_size, self.dicts)
        NER_labels = get_long_tensor(batch[4], batch_size)

        return (words, pos, mask_s, RC_labels, NER_labels)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def gen_labels(sentText, relationMentions, dicts):
    l = len(sentText)
    labels = np.zeros((l,l,len(dicts['label'])), dtype=np.float32)
    for i, rm in enumerate(relationMentions):
        h_e1, h_e2 = sentText.index(rm['em1Text'][0]), sentText.index(rm['em2Text'][0])
        labels[h_e1][h_e2][dicts['label'][rm['label']]] = 1
    return labels
 
def find_index(sentText, en):
    for i in range(len(sentText)):
        if sentText[i:len(en)+i] == en:
            return i, len(en)+i-1

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else 1 for t in tokens] # the id of [UNK] is ``1''
    return ids

def padding_labels(labels, batch_size, dicts):
    """ Convert labels to a padded LongTensor. """
    token_len = max(x.shape[0] for x in labels)
    padded_labels = torch.FloatTensor(batch_size, token_len, token_len, len(dicts['label'])).fill_(0)
    for i, s in enumerate(labels):
        padded_labels[i,:s.shape[0],:s.shape[0]] = torch.FloatTensor(s)
    return padded_labels

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [1 if x != 1 and np.random.random() < dropout \
            else x for x in tokens]

