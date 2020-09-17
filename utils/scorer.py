import torch
import json

BIO_TO_ID = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

def sta(labels_RC, labels_NER, logits_RC, logits_NER, Exact_Match=True):
    labels_num, logits_num, right_num = 0, 0, 0
    # total num of golden relations
    labels_num = labels_RC.sum().item()
    # total num of predicted relations
    logits_RC = torch.where(logits_RC>=0.5, torch.ones_like(logits_RC), torch.zeros_like(logits_RC))
    logits_num = logits_RC.sum().item()
    # total num of predicted right relations
    # right relations
    right_RC = logits_RC.cuda()+labels_RC.cuda()
    right_RC = torch.where(right_RC==2, torch.ones_like(right_RC), torch.zeros_like(right_RC))
    # right entities
    if Exact_Match == True:
        right_EN2RC_mask = get_right_entity_pair(labels_NER, logits_NER)
        right_RC = right_RC * right_EN2RC_mask
    # right num of predicted relations
    right_num = right_RC.sum().item()
    return labels_num, logits_num, right_num

def get_right_entity_pair(labels_NER, logits_NER):
    NER_argmax = torch.argmax(logits_NER, dim=2)
    rp_list = []
    for i in range(NER_argmax.size(0)):
        rp_list.append(find_right(labels_NER[i], NER_argmax[i]))
    assert(len(rp_list) == NER_argmax.size(0))
    ret = torch.zeros((logits_NER.size(0), logits_NER.size(1), logits_NER.size(1), 1))
    for i in range(len(rp_list)):
        if len(rp_list[i]) <= 1:
            continue
        epairs = get_pairs(rp_list[i])
        for ep in epairs:
            ret[i][ep[0]][ep[1]][0] = 1.
            ret[i][ep[1]][ep[0]][0] = 1.
    return ret.cuda()

def find_right(label_NER, logit_NER):
    stack, ner_right = [], []
    # to list
    label_NER = label_NER.cpu().detach().numpy().tolist()
    logit_NER = logit_NER.cpu().detach().numpy().tolist()
    for i, v in enumerate(label_NER):
        if v == BIO_TO_ID['S'] and label_NER[i] == logit_NER[i]:
            ner_right.append(i)
        elif v == BIO_TO_ID['B']:
            stack.append(i)
        elif v == BIO_TO_ID['E']:
            start = stack.pop(0)
            end = i+1
            if label_NER[start:end] == logit_NER[start:end]:
                ner_right.append(i)
    return ner_right

def get_pairs(ens):
    ret = []
    for i in range(len(ens)):
        for j in range(i+1,len(ens)):
            ret.append((ens[i], ens[j]))
    return ret

