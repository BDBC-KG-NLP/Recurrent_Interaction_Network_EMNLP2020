import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model import *

class Trainer(object):
    def __init__(self, args, embedding_matrix=None):
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

# 0: tokens, 1: mask_sent, 2: ote_labels, 3: opn_labels, 4: ts_labels
def unpack_batch(batch):
    for i in range(len(batch)):
        batch[i] = Variable(batch[i].cuda())
    return batch

# 0: tokens, 1: mask_sent, 2: ote_labels, 3: opn_labels, 4: ts_labels
class MyTrainer(Trainer):
    def __init__(self, args):
        self.args             = args
        self.model            = Toy_model(args).cuda()
        self.parameters       = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer        = torch.optim.Adam(self.parameters, lr=args.lr)

    def update(self, batch):
        batch = unpack_batch(batch)
        # step forward
        self.model.train()
        ner_loss, rc_loss, _, _ = self.model(batch)

        # task loss
        loss = ner_loss + rc_loss 

        # backward of task loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.args.max_grad_norm)
        self.optimizer.step()

        # loss value
        ner_loss = ner_loss.item()
        rc_loss  = rc_loss.item()

        return ner_loss, rc_loss 

    def predict(self, batch):
        with torch.no_grad():
            batch = unpack_batch(batch)
            # forward
            self.model.eval()
            ner_loss, rc_loss, ner_pred, rc_pred  = self.model(batch)

            # loss value
            ner_loss = ner_loss.item()
            rc_loss  = rc_loss.item()

        return ner_loss, rc_loss, ner_pred, rc_pred
