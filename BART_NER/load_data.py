import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from functools import cmp_to_key
from itertools import chain
from utils import cmp
import numpy as np

def Load(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        json_result = json.load(f)

    sentences = []
    mner_spans = []
    for item in json_result:
        sentences.append(item['words'])
        mner_span = []
        for dic_spans in item['aspects']:
            mner_span.append((dic_spans['from'], dic_spans['to'],
                           dic_spans['polarity']))
        mner_spans.append(mner_span)
    print('Loading....')
    print('sentences[0]:',sentences[0])
    print('mner_spans[0]:',mner_spans[0])
    print('====================')
    return sentences,mner_spans

def PrepareData(args,sentences,mner_spans):
    #
    tokenizer = AutoTokenizer.from_pretrained(args.bart_path)
    # 注入任务特定token
    mner_token,per_token,org_token,loc_token,other_token = '<<MNER>>','<<PER>>','<<ORG>>','<<LOC>>','<<OTHER>>'
    mapping = {'MNER':'<<MNER>>','PER':'<<PER>>','ORG':'<<ORG>>','LOC':'<<LOC>>','OTHER':'<<OTHER>>'}
    mapping2id = {} # mapping在bart tokenizer中的id
    mapping2targetid = {} # 输出端index
    additional_token = [mner_token,per_token, org_token, loc_token, other_token]
    tokenizer.unique_no_split_tokens += additional_token
    tokenizer.add_tokens(additional_token)

    for key,value in mapping.items():
        key_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value))[0]
        mapping2id[key] = key_id
        mapping2targetid[key] = len(mapping2targetid) + 2

    # 独有，对mner label的处理
    target_shift = len(mapping2targetid) + 2  # index位移
    # 对sentences做bert分词
    input_ids = []
    mner_label = []
    gt_spans = []
    mner_label_masks = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        mner_span = mner_spans[i]
        mner_span_sorted = sorted(mner_span,key = cmp_to_key(cmp))
        word_bpes = [[tokenizer.bos_token_id]] #这里方便下面的cum_sum
        for word in sentence:
            bpes = tokenizer.tokenize(word)
            bpes = tokenizer.convert_tokens_to_ids(bpes)
            word_bpes.append(bpes)
        word_bpes.append([tokenizer.eos_token_id])
        word_bpes_list = list(chain(*word_bpes))
        input_ids.append(word_bpes_list.copy())

        # 将mner label映射到输出端，cumsum便于识别其在子词中的位置
        lens = list(map(len,word_bpes))
        cum_lens = np.cumsum(list(lens)).tolist() #这是因为tokenzier分词后，一个词可能分成多个子词，每个词起始位置就被“拉伸”了，这里使用cumsum累加，可以将[ [],[],[]]中间的列表长度累加起来，记录了拉伸之后的位置
        cur_text = [0,mapping2targetid['MNER'],mapping2targetid['MNER']] # 放在最前面的任务标示token
        mask = [0,0,0]
        gt = []

        for span in mner_span_sorted:
            s_bpe = cum_lens[span[0]] + target_shift
            e_bpe = cum_lens[span[1] - 1] + target_shift #这里是因为数据的标注策略
            polarity = mapping2targetid[span[2]]
            cur_text.extend([s_bpe,e_bpe,polarity])
            gt.append((s_bpe,e_bpe,polarity))
            mask.extend([1,1,1])
        cur_text.append(1)
        mask.append(1)

        mner_label.append(cur_text)
        gt_spans.append(gt)
        mner_label_masks.append(mask)

    # padding
    max_len = max([len(x) for x in input_ids]) # 写得好
    pad_result = torch.full((len(input_ids),max_len),fill_value=tokenizer.pad_token_id)
    mask = torch.zeros(pad_result.size(),dtype = torch.bool)
    for i,x in enumerate(input_ids):
        pad_result[i,:len(x)] = torch.tensor(input_ids[i],dtype=torch.long)
        mask[i,:len(x)] = True
    input_ids = pad_result.clone()
    attention_mask = mask.clone()

    span_max_len = max([len(x) for x in mner_label])
    for i in range(len(mner_label_masks)):
        add_len = span_max_len - len(mner_label_masks[i])
        mner_label_masks[i] = mner_label_masks[i] + [0 for ss in range(add_len)]
        mner_label[i] = mner_label[i] + [1 for ss in range(add_len)]

    mner_label = torch.tensor(mner_label)
    mner_label_masks = torch.tensor(mner_label_masks)
    return input_ids,attention_mask,mner_label,mner_label_masks,gt_spans,mapping2id,tokenizer

class MNER_BART_Dataset(Dataset):
    def __init__(self,args,input_ids,attention_mask,mner_label,mner_label_masks,gt_spans):
        self.args = args
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.mner_label = mner_label
        self.mner_label_masks = mner_label_masks
        self.gt_spans = gt_spans

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, index):
        input_ids = self.input_ids[index].to(self.args.device)
        attention_mask = self.attention_mask[index].to(self.args.device)
        mner_label = self.mner_label[index].to(self.args.device)
        mner_label_masks = self.mner_label_masks[index].to(self.args.device)
        gt_spans = self.gt_spans
        return input_ids,attention_mask,mner_label,mner_label_masks,gt_spans





