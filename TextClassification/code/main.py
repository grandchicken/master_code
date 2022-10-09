import argparse
import torch
from torch.utils.data import DataLoader
from load_data import PreloadData,ClassificationDataset
from model import Sample_Bert_Text_Classification
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', type=str, default='/home/data_ti6_d/lich/pretrain_model/bert_base_chinese')
parser.add_argument('--train_path', type=str, default='../data/train.txt')
parser.add_argument('--dev_path', type=str, default='../data/dev.txt')
parser.add_argument('--test_path', type=str, default='../data/test.txt')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--label_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

print(torch.cuda.is_available())

train_p = PreloadData(args,split = 'train')
train_input_ids,train_attention_masks,train_label_ids,label2id,id2label = train_p.do_process()
train_data = ClassificationDataset(train_input_ids,train_attention_masks,train_label_ids)

dev_p = PreloadData(args,split = 'dev')
dev_input_ids,dev_attention_masks,dev_label_ids,_,_ = dev_p.do_process()
dev_data = ClassificationDataset(dev_input_ids,dev_attention_masks,dev_label_ids)

test_p = PreloadData(args,split = 'test')
test_input_ids,test_attention_masks,test_label_ids,_,_ = test_p.do_process()
test_data = ClassificationDataset(test_input_ids,test_attention_masks,test_label_ids)

train_dataloader = DataLoader(train_data,batch_size=args.batch_size,shuffle= True)
dev_dataloader = DataLoader(dev_data,batch_size=args.batch_size,shuffle= True)
test_dataloader = DataLoader(test_data,batch_size=args.batch_size,shuffle= True)

print(label2id)
model = Sample_Bert_Text_Classification(args).to(args.device)
t = Trainer(args,model,label2id,train_dataloader,dev_dataloader,test_dataloader)
t.train()
