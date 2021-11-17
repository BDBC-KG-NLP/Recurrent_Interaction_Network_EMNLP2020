"""
Joint model for relation classification.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# 0: tokens, 1: pos, 2: mask_s, 3: labels
class Toy_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_matrix = args.embedding_matrix
        self.rounds = args.rounds
        self.encoder = LSTMRelationModel(args)

        # create embedding layers
        self.emb = nn.Embedding(args.vocab_size, args.dim_w, padding_idx=0)
        self.init_embeddings()
        self.emb.weight.requires_grad = True 
        self.pos_emb = nn.Embedding(len(args.pos2id), args.dim_pos, padding_idx=0)

        # dropout
        self.input_dropout = nn.Dropout(args.dropout_rate)

        # GRU
        self.GRUg = torch.nn.GRUCell(5, args.dim_bilstm_hidden*2)
        self.GRUr = torch.nn.GRUCell(len(args.label2id), args.dim_bilstm_hidden*2)

        # classifer
        self.Lr = nn.Linear(4*args.dim_bilstm_hidden, 2*args.dim_bilstm_hidden)
        self.Cr = nn.Linear(2*args.dim_bilstm_hidden, len(args.label2id))
        self.Cg = nn.Linear(2*args.dim_bilstm_hidden, 5)

        # loss function 
        self.ce_loss  = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')

    def init_embeddings(self):
        if self.embedding_matrix is not None:
            self.embedding_matrix = torch.from_numpy(self.embedding_matrix)
            self.emb.weight.data.copy_(self.embedding_matrix)

    # 0: tokens, 1: pos, 2: mask_s, 3: NER_labels, 4: RC_labels
    def forward(self, inputs):
        
        # Bilstm encoder
        tokens, pos, mask_s, NER_labels, RC_labels = inputs
        tokens_embs = self.emb(tokens)
        rnn_inputs = [tokens_embs]
        rnn_inputs += [self.pos_emb(pos)]
        rnn_inputs = torch.cat(rnn_inputs, dim=2)    
        lens = mask_s.sum(dim=1)
        rnn_inputs = self.input_dropout(rnn_inputs)
        H = self.encoder((rnn_inputs, lens.cpu())) 

        # generate mask 
        mask_tmp = mask_s.unsqueeze(1).repeat(1, H.size(1), 1)
        mask_RC  = torch.zeros_like(mask_s.unsqueeze(1).repeat(1, H.size(1), 1))
        len_int  = lens.int()
        for i in range(H.size(0)):
            mask_RC[i, :len_int[i], :len_int[i]] = mask_tmp[i, :len_int[i], :len_int[i]]
        
        # recurrent interaction
        Hg = H
        Hr = H
        for i in range(self.rounds):

            # Cg
            logits_Cg = self.Cg(Hg)
            prob_Cg = F.softmax(logits_Cg, dim=2)

            # Cr 
            e1  = Hr.unsqueeze(2).repeat(1, 1, H.size(1), 1)
            e2  = Hr.unsqueeze(1).repeat(1, H.size(1), 1, 1)
            e12 = torch.cat([e1, e2], dim=3)
            e12 = F.relu(self.Lr(e12), inplace=True)

            prob_Cr = torch.sigmoid(self.Cr(e12))
            prob_Cr = prob_Cr * mask_RC.unsqueeze(-1)
            prob_Cr = prob_Cr.max(dim=2)[0]
        
            # update
            Hg = self.GRUg(prob_Cg.reshape(-1, 5), H.reshape(-1, self.args.dim_bilstm_hidden*2))
            Hr = self.GRUr(prob_Cr.reshape(-1, len(self.args.label2id)), H.reshape(-1, self.args.dim_bilstm_hidden*2))
            Hg = Hg.view(H.size(0), H.size(1), self.args.dim_bilstm_hidden*2)
            Hr = Hr.view(H.size(0), H.size(1), self.args.dim_bilstm_hidden*2)
            H  = H+Hr+Hg

        # use last Hg for classification 
        logits_Cg = self.Cg(Hg)
        # pred and loss
        ner_pred = torch.argmax(logits_Cg, dim=2)
        ner_pred = ner_pred * mask_s.long()
        ner_loss = self.ce_loss(logits_Cg.reshape(-1, logits_Cg.size(-1)), NER_labels.reshape(-1))
        ner_loss = ner_loss.reshape(logits_Cg.size(0), logits_Cg.size(1))
        ner_loss = (ner_loss * mask_s).sum() / ner_loss.size(0)

        # use last Hr for classification 
        e1 = Hr.unsqueeze(2).repeat(1, 1, H.size(1), 1)
        e2 = Hr.unsqueeze(1).repeat(1, H.size(1), 1, 1)
        e12 = torch.cat([e1, e2], dim=3)
        e12 = F.relu(self.Lr(e12), inplace=True)
        logits_Cr = torch.sigmoid(self.Cr(e12))

        # pred and loss
        rc_pred = logits_Cr
        rc_pred = rc_pred * mask_RC.unsqueeze(-1)
        rc_loss = self.bce_loss(logits_Cr.view(-1, len(self.args.label2id)), RC_labels.view(-1, len(self.args.label2id)))
        rc_loss = rc_loss.reshape(logits_Cr.size(0), logits_Cr.size(1), logits_Cr.size(2), -1)
        rc_loss = rc_loss.sum(dim=3)
        rc_loss = (rc_loss * mask_RC).sum() / rc_loss.size(0)

        return ner_loss, rc_loss, ner_pred, rc_pred

# BiLSTM model 
class LSTMRelationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = args.dim_w + args.dim_pos
        self.rnn = nn.LSTM(self.in_dim, args.dim_bilstm_hidden, 1, batch_first=True, dropout=0.0, bidirectional=True)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.dim_bilstm_hidden, 1, True)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        # unpack inputs
        inputs, lens = inputs[0], inputs[1]
        return self.encode_with_rnn(inputs, lens, inputs.size()[0])

# Initialize zero state
def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0




