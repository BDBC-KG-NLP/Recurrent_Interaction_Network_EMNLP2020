import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# 0: tokens, 1: pos, 2: mask_s, 3: labels
class JointClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix
        self.encoder = LSTMRelationModel(args)

        # create embedding layers
        self.emb = nn.Embedding(args.token_vocab_size, args.emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(args.pos_vocab_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None
        self.init_embeddings()

        # dropout
        self.input_dropout = nn.Dropout(args.input_dropout)

        # GRU
        self.GRUg = torch.nn.GRUCell(len(args.dicts['bio']), args.rnn_hidden*2)
        self.GRUr = torch.nn.GRUCell(len(args.dicts['label']), args.rnn_hidden*2)

        # classifer
        self.Lr = nn.Linear(4*args.rnn_hidden, 2*args.rnn_hidden)
        self.Cr = nn.Linear(2*args.rnn_hidden, len(args.dicts['label']))
        self.Cg = nn.Linear(2*args.rnn_hidden, len(args.dicts['bio']))

        # Fn 
        self.logsoft_fn = nn.LogSoftmax(dim=2)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # top N embeddings
        if self.args.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.args.topn < self.args.token_vocab_size:
            print("Finetune top {} word embeddings.".format(self.args.topn))
            self.emb.weight.register_hook(lambda x: keep_partial_grad(x, self.args.topn))
        else:
            print("Finetune all embeddings.")
        
    # 0: tokens, 1: pos, 2: mask_s
    def forward(self, inputs):

        tokens, pos, mask_s = inputs
        tokens_embs = self.emb(tokens)
        rnn_inputs = [tokens_embs]
        if self.args.pos_dim > 0:
            rnn_inputs += [self.pos_emb(pos)]
        rnn_inputs = torch.cat(rnn_inputs, dim=2)    
        lens = mask_s.sum(dim=1)
        rnn_inputs = self.input_dropout(rnn_inputs)
        H = self.encoder((rnn_inputs, lens)) 

        # mask
        s_len = H.size(1)
        mask_NER = mask_s.unsqueeze(-1).repeat(1, 1, len(self.args.dicts['bio']))
        mask_tmp = mask_s.unsqueeze(-1).repeat(1, 1, len(self.args.dicts['label'])) 
        mask_tmp = mask_tmp.unsqueeze(1).repeat(1, s_len, 1, 1)
        mask_RC = torch.zeros_like(mask_tmp)
        real_len = mask_s.sum(dim=1).int()
        for i in range(mask_tmp.size(0)):
            mask_RC[i, :real_len[i], :real_len[i], :] = mask_tmp[i, :real_len[i], :real_len[i], :]
        
        Hg = H
        Hr = H
        for i in range(self.args.rounds):

            # Cg
            logits_Cg = self.Cg(Hg)
            prob_Cg = F.softmax(logits_Cg, dim=2)

            # Cr 
            e1 = Hr.unsqueeze(2).repeat(1, 1, s_len, 1)
            e2 = Hr.unsqueeze(1).repeat(1, s_len, 1, 1)
            e12 = torch.cat([e1, e2], dim=3)
            e12 = F.relu(self.Lr(e12), inplace=True)
            prob_Cr = torch.sigmoid(self.Cr(e12))
            prob_Cr = prob_Cr * mask_RC
            prob_Cr = torch.where(mask_RC==0, torch.zeros_like(prob_Cr)-10e10, prob_Cr)
            prob_Cr = prob_Cr.max(dim=2)[0]
        
            # update
            Hg = self.GRUg(prob_Cg.reshape(-1, len(self.args.dicts['bio'])), H.reshape(-1, self.args.rnn_hidden*2))
            Hr = self.GRUr(prob_Cr.reshape(-1, len(self.args.dicts['label'])), H.reshape(-1, self.args.rnn_hidden*2))
            Hg = Hg.view(H.size(0), H.size(1), self.args.rnn_hidden*2)
            Hr = Hr.view(H.size(0), H.size(1), self.args.rnn_hidden*2)
            H = H+Hr+Hg 

        # final classification 
        logits_Cg = self.Cg(Hg)
        logits_Cg = self.logsoft_fn(logits_Cg)
        logits_Cg = logits_Cg * mask_NER

        e1 = Hr.unsqueeze(2).repeat(1, 1, s_len, 1)
        e2 = Hr.unsqueeze(1).repeat(1, s_len, 1, 1)
        e12 = torch.cat([e1, e2], dim=3)
        e12 = F.relu(self.Lr(e12), inplace=True)
        logits_Cr = torch.sigmoid(self.Cr(e12))
        logits_Cr = logits_Cr *  mask_RC
        
        return logits_Cg, logits_Cr

# BiLSTM model 
class LSTMRelationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = args.emb_dim + args.pos_dim
        self.rnn = nn.LSTM(self.in_dim, args.rnn_hidden, 1, batch_first=True, bidirectional=True)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, 1, True)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        inputs, lens = inputs[0], inputs[1]
        return self.encode_with_rnn(inputs, lens, inputs.size()[0])

def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0




