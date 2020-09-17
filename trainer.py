import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Joint_model import JointClassifier

class Trainer(object):
    def __init__(self, args, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.args,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

# 0: tokens, 1: pos, 2: mask_s, 3: labels
def unpack_batch(batch, cuda=True):
    inputs, labels_RC, labels_NER = batch[0:3], batch[3], batch[4]
    if cuda:
        inputs = [Variable(i.cuda()) for i in inputs]
        labels_RC = Variable(labels_RC.cuda())
        labels_NER = Variable(labels_NER.cuda())
    else:
        inputs = [Variable(i) for i in inputs]
        labels_RC = Variable(labels_RC)
        labels_NER = Variable(labels_NER)
    return inputs, labels_RC, labels_NER

# 0: tokens, 1: pos, 2: mask_s, 3: labels
class MyTrainer(Trainer):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = JointClassifier(args, emb_matrix=emb_matrix)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.loss_NER = nn.NLLLoss(reduction='sum')
        self.loss_RC =  nn.BCELoss(reduction='sum')
        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.parameters, lr=args.lr)

    def update(self, batch):
        inputs, labels_RC, labels_NER = unpack_batch(batch)
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits_NER, logits_RC = self.model(inputs)
        loss_RC = self.loss_RC(logits_RC.view(-1, len(self.args.dicts['label'])), labels_RC.view(-1, len(self.args.dicts['label']))) / logits_RC.size(0)
        loss_NER = self.loss_NER(logits_NER.view(-1, len(self.args.dicts['bio'])), labels_NER.view(-1)) / logits_NER.size(0)
        # backward
        loss = loss_RC + loss_NER
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        # loss value
        loss_RC_val = loss_RC.item()
        loss_NER_val = loss_NER.item()
        return loss_RC_val, loss_NER_val

    def predict(self, batch):
        with torch.no_grad():
            inputs, labels_RC, labels_NER = unpack_batch(batch)
            # sentence length
            mask_s = inputs[2]
            s_len = mask_s.sum(dim=1)
            # forward
            self.model.eval()
            logits_NER, logits_RC = self.model(inputs)
            # loss 
            loss_RC = self.loss_RC(logits_RC.view(-1, len(self.args.dicts['label'])), labels_RC.view(-1, len(self.args.dicts['label']))) / logits_RC.size(0)
            loss_NER = self.loss_NER(logits_NER.view(-1, len(self.args.dicts['bio'])), labels_NER.view(-1)) / logits_NER.size(0)
            loss_RC_val = loss_RC.item()
            loss_NER_val = loss_NER.item()
        return loss_RC_val, loss_NER_val, logits_NER, logits_RC

