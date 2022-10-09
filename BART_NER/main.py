import argparse
# import fitlog
import torch
from torch.utils.data import DataLoader
import os
from load_data import Load,PrepareData,MNER_BART_Dataset
from model import BART_MODEL_MNER
from predict import BARTPredictor
from train import finetune
from evaluate import AESCSpanMetric
parser = argparse.ArgumentParser()
parser.add_argument('--bart_path',type=str,default='/home/data_ti6_d/lich/pretrain_model/bart-base')
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--lr',type=float,default=5e-5)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--dataset',type=str,default='twitter2017/')
parser.add_argument('--data_root',type=str,default='data/')
parser.add_argument('--label_len',type=int)
parser.add_argument('--every_step',type=int,default = 20)
parser.add_argument('--device',type=str,default = 'cuda')
# parser.add_argument('--log_dir',type=str,default='logs/')
args = parser.parse_args()

# log_dir = args.log_dir
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
# fitlog.set_log_dir(log_dir)
# fitlog.add_hyper(args)
# fitlog.add_hyper_in_file(__file__)


train_path = os.path.join(args.data_root,args.dataset,'train.json')
dev_path = os.path.join(args.data_root,args.dataset,'dev.json')
test_path = os.path.join(args.data_root,args.dataset,'test.json')

train_sentences,train_mner_spans = Load(train_path)
dev_sentences,dev_mner_spans = Load(dev_path)
test_sentences,test_mner_spans = Load(test_path)



input_ids,attention_mask,mner_label,mner_label_masks,gt_spans,mapping2id,tokenizer = PrepareData(args,train_sentences,train_mner_spans)
train_dataset = MNER_BART_Dataset(args,input_ids,attention_mask,mner_label,mner_label_masks,gt_spans)
input_ids,attention_mask,mner_label,mner_label_masks,gt_spans,_,_ = PrepareData(args,dev_sentences,dev_mner_spans)
dev_dataset = MNER_BART_Dataset(args,input_ids,attention_mask,mner_label,mner_label_masks,gt_spans)
input_ids,attention_mask,mner_label,mner_label_masks,gt_spans,_,_ = PrepareData(args,test_sentences,test_mner_spans)
test_dataset = MNER_BART_Dataset(args,input_ids,attention_mask,mner_label,mner_label_masks,gt_spans)

train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
dev_dataloader = DataLoader(dev_dataset,batch_size=args.batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

label_ids = list(mapping2id.values())

model = BART_MODEL_MNER(args,tokenizer,label_ids)
decoder = model.decoder
predict_model = BARTPredictor(args,model,decoder)
metric = AESCSpanMetric(eos_token_id,num_labels=len(label_ids),conflict_id=-1)
finetune(args,model,train_dataloader,dev_dataloader,test_dataloader,metric)

# model = BertNER(args)
# model = model.to('cuda')
# t = Trainer(args,model,label2ids,train_dataloader,dev_dataloader,test_dataloader)
# t.train()
if __name__ == '__main__':
    print('label_ids:',label_ids) # [50265, 50266, 50267, 50268, 50269]
    for batch in train_dataloader:
        input_ids, attention_mask, mner_label, mner_label_masks = batch
        print('input_ids:',input_ids)
        print('attention_mask:',attention_mask)
        print('mner_label:',mner_label)
        print('mner_label_masks:',mner_label_masks)
        print('gt_spans:',gt_spans)
        print(tokenizer.bos_token_id) # 0
        print(tokenizer.eos_token_id) # 2
        print(tokenizer.pad_token_id) # 1
        print(tokenizer.convert_tokens_to_ids('<<MNER>>')) # 50265
        print(tokenizer.convert_tokens_to_ids('<<PER>>')) # 50266
        print(tokenizer.convert_tokens_to_ids('<<LOC>>')) # 50268
        print(tokenizer.convert_tokens_to_ids('<<OTHER>>')) # 50269
        print(tokenizer.convert_tokens_to_ids('<<ORG>>')) # 50267
        break
