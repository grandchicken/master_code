import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
import math

class UMT_MNER(nn.Module):
    def __init__(self,args):
        super(UMT_MNER, self).__init__()
        self.args = args
        self.model = BertModel.from_pretrained(args.bert_path)
        self.fc = nn.Linear(768,args.label_len)
        self.dropout = nn.Dropout(0.1)
        self.crf = CRF(args.label_len, batch_first=True)
        self.multihead_attn = Multihead_Attention(args)
        self.v2t = nn.Linear(2048,args.hidden_size)
        self.gate = nn.Linear(args.hidden_size * 2, args.hidden_size)
    def forward(self,input_ids,attention_mask,segment_ids,label_ids, image_feat, image_mask, is_training = True):
        text_output = self.model(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = segment_ids)[0]
        text_output = self.dropout(text_output)
        logits = self.fc(text_output)

        # 下面要进行text和image的注意力交互
        # text2text
        extend_text_mask = attention_mask.unsqueeze(1).unsqueeze(2) # 16 1 1 seqlen
        extend_text_mask = (1.0 - extend_text_mask) * -10000 # 将mask部分变为负无穷大，方便后面计算softmax
        augmented_text = self.multihead_attn(text_output,text_output,extend_text_mask)

        # text2img
        visual_feat = image_feat.view(-1,2048,49).permute(0,2,1) #batch 49 2048
        visual_feat_project = self.v2t(visual_feat) # batch 49 768
        extend_visual_mask = image_mask.unsqueeze(1).unsqueeze(2)
        extend_visual_mask = (1.0 - extend_visual_mask) * -10000.0
        text2image_attention = self.multihead_attn(augmented_text, visual_feat_project, extend_visual_mask)

        # img2text
        visual_feat_project = self.v2t(visual_feat)  # batch 49 768
        image2text_attention = self.multihead_attn(visual_feat_project, augmented_text, extend_text_mask) # batch 49 768

        # text2text
        final_text = self.multihead_attn(augmented_text, image2text_attention, extend_visual_mask)

        # visual gate
        merge_text = torch.cat((text2image_attention,final_text),dim = -1) # batch seqlen 768*2
        gate_value = torch.sigmoid(self.gate(merge_text))
        gated_converted_att_vis_embed = torch.mul(gate_value, text2image_attention)
        final_output = torch.cat((final_text, gated_converted_att_vis_embed), dim=-1) # batch seqlen 768*2

        logits = self.fc(final_output)  # 16 63 13

        if is_training:
            main_loss = - self.crf(logits, label_ids, mask=attention_mask.byte(), reduction='mean')
            return main_loss
        else:
            pred_logits = self.crf.decode(logits, mask=attention_mask.byte())
            return pred_logits


class Multihead_Attention(nn.Module):
    def __init__(self,args):
        super(Multihead_Attention, self).__init__()
        self.args = args

        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads) # 每个头多大
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 一共多大

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(args.attention_probs_dropout_prob)

    def forward(self,former_seq,latter_seq,attention_mask):

        Q = self.query(former_seq) # batch seq_len hidden_size
        K = self.key(latter_seq)
        V = self.value(latter_seq)

        Q = self.transpose_for_scores(Q) # batch head_num seq_len head_size
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        # Q * K.T
        attention_scores = torch.matmul(Q,K.transpose(-1,-2)) # batch head_num seq_len seq_len
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask # 为了后面的softmax

        attention_probs = nn.Softmax(dim = -1)(attention_scores) # # 通过softmax把mask中-无穷部分归0，实现mask的效果
        attention_probs = self.dropout(attention_probs)

        # V
        V_output = torch.matmul(attention_probs, V)  # 16 12 63 64
        V_output = V_output.permute(0, 2, 1, 3).contiguous()  # 16 63 12 64
        new_shape = V_output.size()[:-2] + (self.all_head_size,)  # 16 63 768
        output = V_output.view(*new_shape)  # 重新恢复到原来的形状 16 63 768

        return output

    # 把总的维度按头拆开
    def transpose_for_scores(self, x):
        '''
        a = torch.rand(16,63,768)
        print(a.shape) 16 63 768
        a.size()[:-1] + (12,64) 16 63 12 64
        '''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)