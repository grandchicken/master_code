import torch.nn as nn
from module import VBEncoder
from src.modeling import BertLayerNorm,GeLU

class VisualBert_VQA(nn.Module):
    def __init__(self,args,answer_nums):
        super(VisualBert_VQA, self).__init__()
        self.encoder = VBEncoder(args)

        hid_dim = self.encoder.dim
        self.logit_fc = nn.Sequential(nn.Linear(hid_dim,hid_dim*2),
                                      GeLU(),
                                      BertLayerNorm(hid_dim*2,eps=1e-12),
                                      nn.Linear(hid_dim*2,answer_nums))
        self.logit_fc.apply(self.encoder.model.init_bert_weights)

    def forward(self,input_ids,attention_mask,segment_ids,feats,boxes,visual_segment_ids,v_mask):
        x = self.encoder(input_ids,attention_mask,segment_ids,feats,boxes,visual_segment_ids,v_mask)
        logit = self.logit_fc(x)
        return logit





