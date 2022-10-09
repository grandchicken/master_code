from transformers import BertTokenizer,BertModel
import torch

bert_path = '/home/data_ti6_d/lich/pretrain_model/bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertModel.from_pretrained(bert_path)

# Sample
max_len = 0
text = ['I want to drink water','the drink is a good choice for me','The sun is so hot']
token_list = []
for i in range(len(text)):
    tokens = bert_tokenizer.tokenize(text[i])
    if len(tokens) > max_len:
        max_len = len(tokens)
    token_list.append(tokens)

input_ids = []
input_masks = []
for i in range(len(token_list)):
    current_str = ['[CLS]'] + token_list[i]
    current_str = current_str + ['[SEP]']
    input_id = bert_tokenizer.convert_tokens_to_ids(current_str)

    input_mask = [1] * len(input_id)
    padding = [0] * (max_len + 2 - len(input_id))
    input_id += padding
    input_mask += padding
    input_ids.append(input_id)
    input_masks.append(input_mask)

print('input_ids:',input_ids)
print('input_masks:',input_masks)

input_tensor = torch.tensor(input_ids).to(torch.long)
input_masks = torch.tensor(input_masks).to(torch.long)

print('input_tensor:',input_tensor)
print('input_masks:',input_masks)

output = bert_model(input_ids = input_tensor, attention_mask = input_masks)[0] #pooler_output
print('output:',output.shape)

output = bert_model(input_ids = input_tensor, attention_mask = input_masks)[1]
print('output:',output.shape)


