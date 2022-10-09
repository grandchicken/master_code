from tqdm import tqdm
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
class PreloadData:
    def __init__(self,args,split = 'train'):
        self.args = args
        self.bert_path = args.bert_path
        if split == 'train':
            self.file_path = args.train_path
        elif split == 'dev':
            self.file_path = args.dev_path
        else:
            self.file_path = args.test_path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

    def load_data(self,file_path):
        texts = []
        labels = []
        with open(file_path,'r',encoding = 'utf-8') as f:
            for line in tqdm(f.readlines()):
                line = line.strip('\n')
                text,label = line.split('\t')
                texts.append(text)
                labels.append(label)
        return texts,labels

    def tokenize(self,texts):
        tokens = []
        max_len = 0
        for text in texts:
            token = self.tokenizer.tokenize(text)
            token = ['[CLS]'] + token
            token = token + ['[SEP]']
            if len(token) > max_len:
                max_len = len(token)
            tokens.append(token)

        return tokens,max_len

    def convert2ids(self,tokens,labels,max_len):
        print('convert2_ids...')
        input_ids = []
        attention_masks = []

        label2id = {label:i for i,label in enumerate(set(labels))}
        id2label = {i:label for i,label in enumerate(set(labels))}
        label_ids = [label2id[label] for label in labels]

        for token in tqdm(tokens):
            input_id = self.tokenizer.convert_tokens_to_ids(token)
            input_mask = [1] * len(input_id)
            padding = [0] * (max_len - len(input_id))
            input_id += padding
            input_mask += padding
            assert len(input_id) == len(input_mask) == max_len
            input_ids.append(input_id)
            attention_masks.append(input_mask)

        return input_ids,attention_masks,label_ids,label2id,id2label

    def do_process(self):
        texts,labels = self.load_data(self.file_path)
        tokens,max_len = self.tokenize(texts)
        input_ids,attention_masks,label_ids,label2id,id2label = self.convert2ids(tokens,labels,max_len)
        return input_ids,attention_masks,label_ids,label2id,id2label


class ClassificationDataset(Dataset):

    def __init__(self,input_ids,attention_masks,label_ids):
        self.input_ids = torch.tensor(input_ids).to(torch.long)
        self.attention_masks = torch.tensor(attention_masks).to(torch.long)
        self.label_ids = torch.tensor(label_ids).to(torch.long)

    def __getitem__(self, index):
        return self.input_ids[index],self.attention_masks[index],self.label_ids[index]

    def __len__(self):
        return len(self.input_ids)