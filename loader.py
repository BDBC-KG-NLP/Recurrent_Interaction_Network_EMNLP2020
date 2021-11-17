import json
import random
import torch
import numpy as np

bio2id = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, args, file_path):
        self.batch_size = args.batch_size
        self.args = args
        self.file_path = file_path
        self.word2id = {w:i for i,w in enumerate(args.word_vocab)}

        with open(file_path, 'r') as f:
            self.raw_data = json.load(f)
        self.pos2id, self.label2id = args.pos2id, args.label2id 
        self.data = self.preprocess(self.raw_data)
        self.num_examples = len(self.data)

        # chunk into batches
        self.data = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        print("{} batches created for {}".format(len(self.data), self.file_path))

    def gen_labels(self, sentText, relationMentions):
        l = len(sentText)
        labels = np.zeros((l,l,len(self.args.label2id)), dtype=np.float32)
        for i, rm in enumerate(relationMentions):
            h_e1, h_e2 = sentText.index(rm['em1Text'][0]), sentText.index(rm['em2Text'][0])
            labels[h_e1][h_e2][self.args.label2id[rm['label']]] = 1
        return labels

    def padding_labels(self, labels, batch_size):
        """ Convert labels to a padded LongTensor. """
        token_len = max(x.shape[0] for x in labels)
        padded_labels = torch.FloatTensor(batch_size, token_len, token_len, len(self.label2id)).fill_(0)
        for i, s in enumerate(labels):
            padded_labels[i,:s.shape[0],:s.shape[0]] = torch.FloatTensor(s)
        return padded_labels

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []

        for d in data:
            tokens = d['sentText']
            pos    = d['pos']

            # map to ids 
            tokens = map_to_ids(tokens, self.word2id)
            pos    = map_to_ids(pos, self.pos2id)
            l      = len(tokens)
            
            NER_labels = [bio2id['O'] for _ in range(l)]
            for en in d['en_list']:
                if len(en) == 1:
                    en_index = d['sentText'].index(en[0])
                    NER_labels[en_index] = bio2id['S']
                else:
                    sta, end = find_index(d['sentText'], en)
                    tmp_block = [bio2id['B']] + [bio2id['I'] for _ in range(end-sta-1)] + [bio2id['E']]
                    NER_labels[sta:end+1] = tmp_block 
            
            # sentence mask
            mask_s = [1 for i in range(l)]
            processed += [(tokens, pos, mask_s, (d['sentText'], d['relationMentions']), NER_labels)]

        return processed

    def __len__(self):
        return len(self.data)

    # 0: tokens, 1: mask_s, 2: label
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

        # convert to tensors
        words = get_long_tensor(batch[0], batch_size)
        pos   = get_long_tensor(batch[1], batch_size)
        
        # mask_s to tensors 
        mask_s = get_float_tensor(batch[2], batch_size)

        # RC_labels to tensors
        RC_labels = [self.gen_labels(sentText, relationMentions) for sentText, relationMentions in batch[3]]
        RC_labels = self.padding_labels(RC_labels,  batch_size)

        # NER_labels to tensors
        NER_labels = get_long_tensor(batch[4], batch_size)
        
        return [words, pos, mask_s, NER_labels, RC_labels] 

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def find_index(sentText, en): # [sta, end]
    for i in range(len(sentText)):
        if sentText[i:len(en)+i] == en:
            return i, len(en)+i-1

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else vocab['<UNK>'] for t in tokens]
    return ids

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

