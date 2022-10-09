import pickle
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import BertTokenizer

def load(text_path,visual_path):
    df = pd.read_csv(text_path,index_col=0)
    v_feats = pickle.load(open(visual_path, 'rb'))
    choice_df = list(df.columns[1:-2])
    return df,v_feats,choice_df

def preprocess(args,df,split):
    bert_path = args.bert_path
    dataset = df[df['split'] == split]
    id_list = dataset.id.tolist()
    text_list = dataset.TXT.tolist()
    choice_df = dataset.columns[1:-2]
    target_list = dataset[choice_df].to_numpy().astype(np.float32)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    max_len = 0
    tokenss = []
    for i,sent in enumerate(text_list):
        tokens = tokenizer.tokenize(sent)
        tokens = ['[CLS]'] + tokens
        tokens = tokens + ['[SEP]']
        if len(tokens) > max_len:
            max_len = len(tokens)
        tokenss.append(tokens)
    input_ids = []
    attention_mask = []
    segment_ids = []
    for i,tokens in enumerate(tokenss):
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_id)
        padding = (max_len - len(input_id)) * [0]
        segment_id = [0] * len(input_id)
        input_id += padding
        input_mask += padding
        segment_id += padding
        input_ids.append(input_id)
        attention_mask.append(input_mask)
        segment_ids.append(segment_id)
    return input_ids,attention_mask,segment_ids,id_list,target_list


class OpenIDataset(Dataset):
    def __init__(self,input_ids,attention_mask,segment_ids,id_list,target_list,v_feats):
        self.input_ids = torch.tensor(input_ids).to(torch.long)
        self.attention_mask = torch.tensor(attention_mask).to(torch.long)
        self.segment_ids = torch.tensor(segment_ids).to(torch.long)
        self.v_feats = v_feats
        self.id_list = id_list
        self.target_list =  target_list

    def __len__(self):
        return len(self.id_list)
    def __getitem__(self, index):
        id_ = self.id_list[index]
        choice = self.target_list[index]
        boxes,feats,(img_w,img_h) = self.v_feats[id_]
        visual_segment_ids = torch.ones(feats.shape[0]).to(torch.long)
        v_mask = torch.ones(feats.shape[0]).to(torch.long) #36
        segment_ids = self.segment_ids[index]
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]

        return id_,input_ids,attention_mask,segment_ids,boxes,feats,visual_segment_ids,v_mask,choice

if __name__ == '__main__':
    text_path = '/home/data_ti6_d/lich/program_sample/visualbert VQA/data/openIdf.csv'
    visual_path = '/home/data_ti6_d/lich/program_sample/visualbert VQA/data/openI_visual_features.pickle'
    df, v_feats, choice_df = load(text_path,visual_path)
    # print(v_feats) # [2048*36 n个]

    a = OpenIDataset(df,v_feats,'test')
    print(a.text_list)
    # print(a.target_list.shape) # 772 15