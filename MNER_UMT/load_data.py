import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer
import os
import numpy as np
# 读数据
def read_data(file_path):
    texts = []
    labels = []
    image_ids = []
    with open(file_path,'r',encoding='utf-8') as f:
        for i ,line in tqdm(enumerate(f.readlines())):
            line = line.strip()
            if line == '':
                pass
            if i % 3 == 0:
                texts.append(line.split(' '))
            elif i % 3 == 1:
                labels.append(line.split(' '))
            else:
                image_ids.append(line)
        assert len(texts) == len(labels)
    print('Sample Texts:',texts[0])
    print('Sample Labels:',labels[0])
    print('Sample Image ids:',image_ids[0])
    return texts,labels,image_ids

# 建立label与id的映射，注意这里添加了'X'
def generate_label_map(train_labels,dev_labels,test_labels):
    total_labels = train_labels + dev_labels + test_labels
    total_labels = [l for label in total_labels for l in label]
    label_set = set(total_labels)
    label_set.add('[PAD]')
    label_set.add('[CLS]')
    label_set.add('[SEP]')
    label_set.add('X')
    label_len = len(label_set)
    label2ids = {label:i for i,label in enumerate(label_set)}
    ids2label = {i:label for i,label in enumerate(label_set)}
    return label_len,label2ids,ids2label

def preprocess_data(args,texts,labels,label2ids):
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    tokenss = []
    new_labels = []
    for i,text in enumerate(tqdm(texts)):
        label = labels[i]
        tokens = []
        new_label = []
        for j,word in enumerate(text):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_temp = label[j]

            for m in range(len(token)):
                if m == 0:
                    new_label.append(label_temp)
                else:
                    new_label.append('X') #对于bert切分过程中一个词拆解为多个子词的现象，将子词标为X

        tokens = ['[CLS]'] + tokens
        tokens = tokens + ['[SEP]']
        tokenss.append(tokens)

        new_label = ['[CLS]'] + new_label
        new_label = new_label + ['[SEP]']
        new_labels.append(new_label)

    max_len = 0
    for i,tokens in enumerate(tokenss):
        if len(tokens) > max_len:
            max_len = len(tokens)

    attention_mask = []
    input_ids = []
    segment_ids = []
    label_ids = []
    for i,tokens in enumerate(tokenss):
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        label_id = [label2ids[label] for label in new_labels[i]]
        mask = [1] * len(tokens)
        segment_id = [1] * len(tokens)
        padding = [0] * (max_len - len(tokens))
        padding_label = ['[PAD]'] * (max_len - len(tokens))
        padding_label_id = [label2ids[label] for label in padding_label]

        input_id += padding
        mask += padding
        segment_id += padding
        label_id += padding_label_id

        input_ids.append(input_id)
        attention_mask.append(mask)
        segment_ids.append(segment_id)
        label_ids.append(label_id)

    return input_ids,attention_mask,segment_ids,label_ids

def get_image_feats(args,image_ids):
    image_feats = []
    image_root = args.image_root

    for image_id in image_ids:
        image_id = image_id.split('.')[0] + '.npy'
        image_path = os.path.join(image_root,image_id)
        image_feat = np.load(image_path)
        image_feats.append(image_feat)
    return image_feats

class NER_Dataset(Dataset):
    def __init__(self,args,input_ids,attention_mask,segment_ids,label_ids,image_feats):
        self.args = args
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.image_feats = image_feats

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_id = torch.tensor(self.input_ids[index]).to(torch.long)
        attention_mask = torch.tensor(self.attention_mask[index]).to(torch.long)
        segment_id = torch.tensor(self.segment_ids[index]).to(torch.long)
        label_id = torch.tensor(self.label_ids[index]).to(torch.long)
        image_feat = torch.tensor(self.image_feats[index])
        image_mask = torch.ones(49).to(torch.long)
        return input_id,attention_mask,segment_id,label_id,image_feat,image_mask


