import argparse
import fitlog

from torch.utils.data import DataLoader
import os
from load_data import read_data,generate_label_map,preprocess_data,NER_Dataset,get_image_feats
from model import UMT_MNER
from train import Trainer
from config import add_modelparameter

parser = argparse.ArgumentParser()
parser.add_argument('--bert_path',type=str,default='/home/data_ti6_d/lich/pretrain_model/bert-base-cased')
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--lr',type=float,default=5e-5)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--dataset',type=str,default='twitter2017/')
parser.add_argument('--data_root',type=str,default='data/')
parser.add_argument('--label_len',type=int)
parser.add_argument('--every_step',type=int,default = 20)
parser.add_argument('--log_dir',type=str,default='logs/')
parser.add_argument('--image_root',type=str,default='tw2017embedding/')
parser.add_argument('--task',type=str,default='2017')
parser = add_modelparameter(parser)

args = parser.parse_args()

if args.task == '2015':
    args.image_root =  'tw2015embedding/'
    args.dataset = 'twitter2015/'
if args.task == '2017':
    args.image_root = 'tw2017embedding/'
    args.dataset = 'twitter2017/'


log_dir = args.log_dir
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
fitlog.set_log_dir(log_dir)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)


train_path = os.path.join(args.data_root,args.dataset,'train.txt')
dev_path = os.path.join(args.data_root,args.dataset,'dev.txt')
test_path = os.path.join(args.data_root,args.dataset,'test.txt')

train_texts,train_labels,train_imageids = read_data(train_path)
dev_texts,dev_labels,dev_imageids = read_data(dev_path)
test_texts,test_labels,test_imageids = read_data(test_path)

train_image_feats = get_image_feats(args,train_imageids)
dev_image_feats = get_image_feats(args,dev_imageids)
test_image_feats = get_image_feats(args,test_imageids)

label_len,label2ids,ids2label = generate_label_map(train_labels,dev_labels,test_labels)
args.label_len = label_len

input_ids,attention_mask,segment_ids,label_ids = preprocess_data(args,train_texts,train_labels,label2ids)
train_dataset = NER_Dataset(args,input_ids,attention_mask,segment_ids,label_ids,train_image_feats)
input_ids,attention_mask,segment_ids,label_ids = preprocess_data(args,dev_texts,dev_labels,label2ids)
dev_dataset = NER_Dataset(args,input_ids,attention_mask,segment_ids,label_ids,dev_image_feats)
input_ids,attention_mask,segment_ids,label_ids = preprocess_data(args,test_texts,test_labels,label2ids)
test_dataset = NER_Dataset(args,input_ids,attention_mask,segment_ids,label_ids,test_image_feats)

train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
dev_dataloader = DataLoader(dev_dataset,batch_size=args.batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

model = UMT_MNER(args)
model = model.to('cuda')
t = Trainer(args,model,label2ids,train_dataloader,dev_dataloader,test_dataloader)
t.train()
