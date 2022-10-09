from transformers import BertModel
import torch.nn as nn
import torch
class Sample_Bert_Text_Classification(nn.Module):
    def __init__(self,args):
        super(Sample_Bert_Text_Classification, self).__init__()
        self.bert_model = BertModel.from_pretrained(args.bert_path)
        self.fc1 = nn.Linear(768,100)
        self.fc2 = nn.Linear(100,args.label_len)
    def forward(self,batch_input_ids,batch_attention_masks):
        output = self.bert_model(input_ids = batch_input_ids,attention_mask = batch_attention_masks)[0]
        output = torch.mean(output, dim=1)
        output = self.fc1(output)
        output = self.fc2(output)

        return output