import torch
import torch.nn as nn
from src.optimization import BertAdam
import numpy as np
from tqdm import tqdm
class Trainer:
    def __init__(self,args,model,train_dataloader,test_dataloader,choice_df):
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.sgmd = torch.nn.Sigmoid()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.findings = choice_df
        self.optim = BertAdam(list(model.parameters()), lr=args.lr, warmup=0.1)

    def train(self):
        for e in range(self.args.epochs):
            epoch_loss = 0
            for i,batch in enumerate(self.train_dataloader):
                id_, input_ids, attention_mask, segment_ids, boxes, feats, visual_segment_ids, v_mask, choice = batch
                self.model.train()
                self.optim.zero_grad()
                input_ids,attention_mask,segment_ids,boxes,feats,visual_segment_ids,v_mask,choice = \
                    input_ids.cuda(),attention_mask.cuda(),segment_ids.cuda(),boxes.cuda(),feats.cuda(),visual_segment_ids.cuda(),v_mask.cuda(),choice.cuda()
                logits = self.model(input_ids,attention_mask,segment_ids,feats,boxes,visual_segment_ids,v_mask)
                running_loss = self.loss(logits,choice)
                running_loss = running_loss * logits.size(1)
                epoch_loss += running_loss
                running_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.)
                self.optim.step()
            print("Epoch "+str(e)+ ":Training Loss " + str(epoch_loss/len(self.train_dataloader)))
            print("Evaluation: ")
            self.eval()
    def eval(self):
        self.model.eval()
        logit_list = []
        target_list = []
        for i,batch in enumerate(self.test_dataloader):
            id_, input_ids, attention_mask, segment_ids, boxes, feats, visual_segment_ids, v_mask, choice = batch
            input_ids, attention_mask, segment_ids, boxes, feats, visual_segment_ids, v_mask = \
                input_ids.cuda(), attention_mask.cuda(), segment_ids.cuda(), boxes.cuda(), feats.cuda(), visual_segment_ids.cuda(), v_mask.cuda()
            target_list.append(choice.numpy().tolist())
            with torch.no_grad():
                logit = self.model(input_ids,attention_mask,segment_ids,feats,boxes,visual_segment_ids,v_mask)
                logit_list.append(self.sgmd(logit).cpu().numpy())
        target_array = np.concatenate(target_list,axis = 0)
        logit_array = np.concatenate(logit_list,axis = 0)

        acc_list = []
        for i,d in enumerate(self.findings[:-1]):
            acc = np.mean(target_array[:,i] == (logit_array[:,i] >= 0.5))
            print(i,d,acc)
            acc_list.append(acc)
        print("Averaged: "+ str(np.average(acc_list)))


