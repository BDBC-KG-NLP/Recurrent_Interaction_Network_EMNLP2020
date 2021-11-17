import os
import shutil
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import pickle
import copy
import json

from config import *
from evals import *
from loader import DataLoader 
from trainer import MyTrainer

seed = random.randint(1, 10000)

class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)
    
    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message, file=out)


# load vocab and embedding matrix
dataset_path          = "./data/%s"        % (args.dataset)
vocab_path            = "%s/vocab.pkl"        % dataset_path
embedding_path        = "%s/embedding.npy"    % dataset_path
print('loading vocab and embedding matrix from {}'.format(dataset_path))
with open(vocab_path, 'rb') as f:
    word_vocab = pickle.load(f)
args.word_vocab = word_vocab
embedding_matrix = np.load(embedding_path)
args.embedding_matrix = embedding_matrix
assert embedding_matrix.shape[0] == len(word_vocab)
assert embedding_matrix.shape[1] == args.dim_w
print('size of vocab: {}'.format(len(word_vocab)))
print('shape of loaded embedding matrix: {}'.format(embedding_matrix.shape))
args.vocab_size = len(word_vocab)

# load data
train_path  = '%s/train.json' % (dataset_path)
test_path   = '%s/test.json'  % (dataset_path)

# generate pos2id, label2id
print("Generating pos2id and label2id ...")
with open(train_path, 'r') as f:
    raw_train = json.load(f)
with open(test_path, 'r') as f:
    raw_test = json.load(f)
raw_data = raw_train + raw_test
pos_list, label_list = ['<PAD>'], []

for d in raw_data:
    for rm in d['relationMentions']:
        if rm['label'] not in label_list:
            label_list.append(rm['label'])
    for p in d['pos']:
        if p not in pos_list:
            pos_list.append(p)

pos2id   = {p:i for i,p in enumerate(pos_list)}
label2id = {l:i for i,l in enumerate(label_list)}
args.pos2id   = pos2id
args.label2id = label2id

print("Loading data from {} with batch size {}...".format(dataset_path, args.batch_size))
train_batches  = DataLoader(args, train_path)
test_batches   = DataLoader(args, test_path)

# create the folder for saving the best model
if os.path.exists(args.save_dir) != True:
    os.mkdir(args.save_dir)

log_file = FileLogger(args.save_dir+"/log.txt")

print('Building model...')
# create model
trainer_rin  = MyTrainer(args)

# start training
estop      = 0
batch_num  = len(train_batches)
current_best_scores = [-1, -1, -1]
for epoch in range(1, args.n_epoch+1):
    
    if estop > args.early_stop:
        break

    train_ner_loss, train_rc_loss, train_step = 0., 0., 0
    for i in range(batch_num):
        batch = train_batches[i]
        ner_loss, rc_loss = trainer_rin.update(batch)
        train_ner_loss += ner_loss
        train_rc_loss  += rc_loss
        train_step += 1
        
        # print training loss 
        if train_step % args.print_step == 0:
            print("[{}] train_ner_loss: {:.4f}, train_rc_loss: {:.4f}".format(epoch, train_ner_loss/train_step, train_rc_loss/train_step))
    
    # evaluate on unlabel set
    print("")
    print("Evaluating...Epoch: {}".format(epoch))
    eval_scores, eval_ner_loss, eval_rc_loss = evaluate_program(trainer_rin, test_batches, args.exact_match)
    print("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))
    # loging
    log_file.log("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))

    if eval_scores[-1] > current_best_scores[-1]:
        current_best_scores = eval_scores
        trainer_rin.save(args.save_dir+'/best_model.pt')
        print("New best model saved!")
        log_file.log("New best model saved!")
        estop = 0

    estop += 1
    print("")


print("Training ended with {} epochs.".format(epoch))

# final results
trainer_rin.load(args.save_dir+'/best_model.pt')
eval_scores, eval_ner_loss, eval_rc_loss = evaluate_program(trainer_rin, test_batches, args.exact_match)

print("Final result:")
print("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))

# loging
log_file.log("Final result:")
log_file.log("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))
