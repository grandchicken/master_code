from load_data import load,OpenIDataset,preprocess

from torch.utils.data import DataLoader
import argparse
from model import VisualBert_VQA
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--visual_path',type=str,default='/home/data_ti6_d/lich/program_sample/visualbert VQA/data/openI_visual_features.pickle')
parser.add_argument('--text_path',type=str,default='/home/data_ti6_d/lich/program_sample/visualbert VQA/data/openIdf.csv')
parser.add_argument('--bert_path',type=str,default='/home/data_ti6_d/lich/pretrain_model/bert-base-uncased')
parser.add_argument('--pretrained',type=str,default='/home/data_ti6_d/lich/pretrain_model/visualbert.th')
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--lr',type=float,default=1e-5)
args = parser.parse_args()


text_path = args.text_path
visual_path = args.visual_path
df, v_feats, choice_df = load(text_path, visual_path)
train_input_ids,train_attention_mask,train_segment_ids,train_id_list,train_target_list = preprocess(args,df,split = 'train')
test_input_ids,test_attention_mask,test_segment_ids,test_id_list,test_target_list = preprocess(args,df,split = 'test')

train_dataset = OpenIDataset(train_input_ids,train_attention_mask,train_segment_ids,train_id_list,train_target_list,v_feats)
test_dataset = OpenIDataset(test_input_ids,test_attention_mask,test_segment_ids,test_id_list,test_target_list,v_feats)
print(list(train_dataset.v_feats.values())[0][1].shape) # 36 2048
train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size,shuffle = True)
test_dataloader = DataLoader(test_dataset,batch_size = args.batch_size,shuffle = True)

model = VisualBert_VQA(args,answer_nums = len(choice_df))
model.encoder.load(args.pretrained)
model = model.to('cuda')

t = Trainer(args,model,train_dataloader,test_dataloader,choice_df)
t.train()



