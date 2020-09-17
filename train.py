import os
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import pickle

from utils import helper
from loader import DataLoader
from trainer import MyTrainer
from utils.scorer import sta

parser = argparse.ArgumentParser()

# nyt
parser.add_argument('--dataset', type=str, default='nyt', help='Dataset directory')
parser.add_argument('--emb_dim', type=int, default=100, help='Word embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=10, help='POS embedding dimension.')
parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
parser.add_argument('--input_dropout', type=float, default=0.1, help='input dropout rate.')
# 7 for Exact Match, 4 for Partial Match
parser.add_argument('--rounds', type=int, default=4, help='Number of rounds.')
parser.add_argument('--log_step', type=int, default=100, help='Print log every k steps.')

# webnlg
'''
parser.add_argument('--dataset', type=str, default='webnlg', help='Dataset directory')
parser.add_argument('--emb_dim', type=int, default=100, help='Word embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=10, help='POS embedding dimension.')
parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
parser.add_argument('--input_dropout', type=float, default=0.1, help='input dropout rate.')
# 3 for Exact Match, 2 for Partial Match
parser.add_argument('--rounds', type=int, default=2, help='Number of rounds.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
'''

# shared hyperparameters for all datasets
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', default=False, help='Lowercase all words.')
parser.add_argument('--lr', type=float, default=1e-3, help='Applies to sgd and adagrad.')
parser.add_argument('--Exact_Match', default=False, help='Exact Match or Partial Match.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--num_epoch', type=int, default=200, help='Number of total training epochs.')
parser.add_argument('--early_stop', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# load contants
dicts = eval(open('./dataset/'+args.dataset+'/constant.py', 'r').read())
vocab_file = './dataset/'+args.dataset+'/vocab.pkl'
token_vocab = dict()
with open(vocab_file, 'rb') as infile:
    token_vocab['i2w'] = pickle.load(infile)
    token_vocab['w2i'] = {token_vocab['i2w'][i]:i for i in range(len(token_vocab['i2w']))}

emb_file = './dataset/'+args.dataset+'/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == len(token_vocab['i2w'])
assert emb_matrix.shape[1] == args.emb_dim

args.token_vocab_size = len(token_vocab['i2w'])
args.pos_vocab_size = len(dicts['pos'])

dicts['token'] = token_vocab['w2i']
args.dicts = dicts

# load training set and test set
print("Loading data from {} with batch size {}...".format(args.dataset, args.batch_size))
train_batch = DataLoader('./dataset/'+args.dataset+'/train.json', args.batch_size, args, dicts)
test_batch = DataLoader('./dataset/'+args.dataset+'/test.json', args.batch_size, args, dicts)

# create the folder for saving the best models and log file
model_save_dir = args.save_dir
helper.ensure_dir(model_save_dir, verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + args.log, header="# epoch\ttrain_loss\ttrain_loss_er\ttrain_loss_rc\tdev_loss\tdev_loss_er\tdev_loss_rc\tP\tR\tF1")

print('Building model...')
trainer = MyTrainer(args, emb_matrix=emb_matrix)

# start training
e_stop = 0
train_loss_his, train_loss_ner_his, train_loss_rc_his, test_loss_his, test_loss_ner_his, test_loss_rc_his, P_his, R_his, F1_his = [], [], [], [], [], [], [], [], []
for epoch in range(1, args.num_epoch+1):
    e_stop += 1
    if e_stop > args.early_stop:
        break

    train_loss, train_loss_ner, train_loss_rc, train_step = 0., 0., 0., 0
    for i, batch in enumerate(train_batch):
        loss_RC, loss_NER = trainer.update(batch)
        loss = loss_RC + loss_NER
        train_loss += loss
        train_loss_ner += loss_NER
        train_loss_rc += loss_RC
        train_step += 1
        if train_step % args.log_step == 0:
            print("train_loss: {:.4f} (ER: {:.4f}, RC: {:.4f})".format(train_loss/train_step, train_loss_ner/train_step, train_loss_rc/train_step))
    
    # evaluating
    print("Evaluating...")
    golden_nums, predict_nums, right_nums = 0, 0, 0
    test_loss, test_loss_ner, test_loss_rc, test_step = 0., 0., 0., 0
    for i, batch in enumerate(test_batch):
        loss_RC, loss_NER, logits_NER, logits_RC = trainer.predict(batch)
        tmp_g, tmp_p, tmp_r = sta(batch[3], batch[4], logits_RC, logits_NER, args.Exact_Match)
        golden_nums += tmp_g
        predict_nums += tmp_p
        right_nums += tmp_r
        loss = loss_RC + loss_NER
        test_loss += loss
        test_loss_ner += loss_NER
        test_loss_rc += loss_RC
        test_step += 1
    if predict_nums == 0:
        P = 0.
    else:
        P = float(right_nums) / predict_nums
    R = float(right_nums) / golden_nums
    if P+R == 0:
        F1 = 0.
    else:
        F1 = 2*P*R/(P+R)
    print("trian_loss: {:.4f} (ER: {:.4f}, RC: {:.4f}), test_loss: {:.4f} (ER: {:.4f}, RC: {:.4f}), P, R, F1: [{:.4f}, {:.4f}, {:.4f}]".format( \
        train_loss/train_step, train_loss_ner/train_step, train_loss_rc/train_step, \
        test_loss/test_step, test_loss_ner/test_step, test_loss_rc/test_step, \
        P, R, F1))
    file_logger.log("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format( \
        epoch, train_loss/train_step, train_loss_ner/train_step, train_loss_rc/train_step, \
        test_loss/test_step, test_loss_ner/test_step, test_loss_rc/test_step, \
        P, R, F1))
    
    train_loss_his.append(train_loss/train_step)
    train_loss_ner_his.append(train_loss_ner/train_step)
    train_loss_rc_his.append(train_loss_rc/train_step)
    test_loss_his.append(test_loss/test_step)
    test_loss_ner_his.append(test_loss_ner/test_step)
    test_loss_rc_his.append(test_loss_rc/test_step)
    P_his.append(P)
    R_his.append(R)
    
    # save best model
    if epoch == 1 or F1 > max(F1_his):
        trainer.save(args.save_dir + '/best_model.pt')
        print("new best model saved.")
        print("")
        file_logger.log("new best model saved at epoch {}: {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}"\
            .format(epoch, train_loss/train_step, test_loss/test_step, \
            P, R, F1))
        e_stop = 0

    F1_his.append(F1)
    
print("Training ended with {} epochs.".format(epoch))
bt_F1 = max(F1_his)
print("best results: [{:.4f}, {:.4f}, {:.4f}]".format(P_his[F1_his.index(bt_F1)], R_his[F1_his.index(bt_F1)], bt_F1))

