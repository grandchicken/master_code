import torch
import torch.nn as nn
from src.modeling import VBFeatureExtraction as VBE

class VBEncoder(nn.Module):
    def __init__(self,args):
        super(VBEncoder, self).__init__()
        self.args = args
        self.model = VBE.from_pretrained(args.bert_path)
    @property
    def dim(self):
        return 768
    def forward(self,input_ids,attention_mask,segment_ids,feats,boxes,visual_segment_ids,v_mask):
        output =  self.model(input_ids = input_ids,token_type_ids = segment_ids,attention_mask = attention_mask,
                             visual_feats = feats, visual_token_type_ids = visual_segment_ids,visual_attention_mask = v_mask)
        return output

    def load(self,path):
        state_dict = torch.load(path)
        print('load pretrained model from %s'%path)
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print('Weights in loaded but not in model:')
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()
        self.model.load_state_dict(state_dict,strict=False)
