import torch.nn as nn
import torch
from transformers import BartModel,BartTokenizer
from utils import Span_Loss
import torch.nn.functional as F
# BART
# last_hidden_state : Sequence of hidden-states at the output of the last layer of the decoder of the model
class BART_MODEL_MNER(nn.Module):
    def __init__(self,args,tokenizer,label_ids):
        super(BART_MODEL_MNER, self).__init__()
        model = BartModel.from_pretrained(args.bart_path)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape #返回词表大小，768，获得bart词表大小
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens) + num_tokens) #因为注入了新词，因此要resize词表，不然报错
        model.encoder.embed_tokens.padding_idx = args.pad_token_id

        # 由于注入了新词(<<MNER>>,<<LOC>>)，这些bart本身没有学习，因此需要对新词的embedding进行初始化
        _tokenizer = BartTokenizer.from_pretrained(args.bart_path)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index) > 1:
                    raise RuntimeError('wrong split')
                else:
                    index = index[0]

                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2])) #把<<>>删掉，看原来的意思，以它原意来初始化
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                # 若存在子词
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        self.encoder = model.encoder
        # causal_mask
        causal_mask = torch.zeros(512,512).fill_(float('-inf')) # 512是编码器编码最大长度
        causal_mask = causal_mask.triu(diagonal=1) #取上对角线
        # decoder
        decoder = model.decoder
        self.decoder = BartDecoder(args,decoder,tokenizer,label_ids,causal_mask) #传tokenizer是因为tokenizer被我们修改了
        # loss
        self.loss = Span_Loss()

    def forward(self,input_ids,attention_mask,mner_label,mner_label_masks,first = None):
        output_dict = self.encoder(input_ids = input_ids,attention_mask = attention_mask,
                              return_dict = True, output_hidden_states = True)
        output = output_dict.last_hidden_state
        src_embed_outputs = output_dict.hidden_state[0] #是啥（是hidden的第一层）？ first又是啥？
        state = BartState(output,attention_mask,input_ids,src_embed_outputs,first) #state以及编码了input_ids 和attention mask

        logits = self.decoder(mner_label,state)
        loss = self.span_loss_fct(mner_label[:,1:],logits,mner_label_masks[:,1:]) #这里计算loss，去掉了任务标示token
        return loss

class State:
    def __init__(self,encoder_output = None, encoder_mask = None, **kwargs):
        '''
        每个decoder都有对应的State对象来承载encoder的输出以及当前时刻之前的decoder状态
        :param encoder_output:
        :param encoder_mask:
        :param kwargs:
        '''
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0
    @property
    def num_samples(self):
        # 返回state中包含的是多少个sample的encoder状态，主要用于Generate的时候确定batch的大小
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None
    @property
    def decode_length(self):
        # 当前decode到哪个token了，decode只会从decode_length 之后的token开始decode，为0说明还没开始decode
        return self._decode_length

    @decode_length.setter
    def decode_length(self,value):
        self._decode_length = value

    def reorder_state(self,indices):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask,indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output,indices)

    def _reorder_state(self,state,indices,dim = 0):
        # 这里我们要求state就是tensor
        state = state.index_select(index = indices,dim = dim)
        return state


class BartState(State):
    def __init__(self,encoder_output,attention_mask,src_tokens,src_embed_outputs,first):
        super(BartState, self).__init__(encoder_output,attention_mask) # 父类初始化
        self.past_key_values = None # ?这是啥
        self.src_tokens = src_tokens
        self.src_embed_outputs = src_embed_outputs
        self.first = first
    def reorder_state(self,indices):
        super(BartState, self).reorder_state(indices)
        self.input_ids = self._reorder_state(self.input_ids,indices) # 根据indices重排input_ids

class BartDecoder(nn.Module):
    def __init__(self,args,decoder,tokenizer,label_ids,causal_mask):
        '''

        :param args:
        :param tokenizer: tokenizer发生了变化，我们注入了自己的token，因此要特地传入
        :param label_ids: ？
        :param causal_mask: 解码时要一步步解
        :param need_tag: ？
        '''
        super(BartDecoder, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.label_ids = label_ids
        self.decoder = decoder
        self.label_start_id = min(label_ids) # label初始是从哪里开始的
        self.label_end_id = max(label_ids) + 1
        self.causal_mask = causal_mask
        self.pad_token_id = tokenizer.pad_token_id
        self.register_buffer('causal_masks',causal_mask.float()) #该方法的作用是定义一组参数，该组参数的特别之处在于：模型训练时不会更新
        # （即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。

        # self.need_tag = need_tag

        mapping = torch.LongTensor([0,2] + label_ids) #[0,2]是啥？
        self.register_buffer('mapping',mapping)
        self.src_start_index = len(mapping)
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self,tokens,state):
        # tokens (batch_size, max_len)
        src_tokens = state.src_tokens
        encoder_pad_mask = state.attention_mask
        encoder_output = state.encoder_output
        # 下面两行的目的是找到1（也就是代表eos的位置)，让1后面的标示为mask，前面的不标示为mask
        '''
        >>> mask = torch.tensor([[0,2,1,1,1],[0,2,3,1,1]])
        >>> c = mask.eq(1).flip(dims = [1]).cumsum(dim=-1)
        >>> tgt = c.flip(dims=[1]).ne(c[:,-1:])
        >>> tgt
        tensor([[False, False, False,  True,  True],
                [False, False, False, False,  True]])
        '''
        cumsum = tokens.eq(1).flip(dims = [1]).cumsum(dim = -1) # 需要print看一下
        tgt_pad_mask = cumsum.flip(dims = [1]).ne(cumsum[:,-1:])# 需要print看一下 FFFFTTTT T代表mask

        mapping_token_mask = tokens.lt(self.src_start_token) #这里返回为1的意味着要从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src)) #不从mapping中（也就是不是我们自己注入的那一部分）取index的暂时用mask盖住
        tag_mapped_token = self.mapping[mapped_tokens] #对没被盖住的做一下映射

        src_tokens_index = tokens - self.src_start_index # 之前作为target shift 加过，现在减回去
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0),0) # 把那些我们注入的那一部分盖住
        word_mapped_tokens = src_tokens.gather(index = src_tokens_index,dim = 1)

        tokens = tokens.where(mapping_token_mask,tag_mapped_token,word_mapped_tokens) #根据是否从mapping取index的条件，将两部分分别映射为id
        tokens = tokens.masked_fill(tgt_pad_mask,self.pad_token_id) #将原本就是pad的部分盖住

        if self.training:
            tokens = tokens[:, :-1] # 最后一个词不取？
            decoder_pad_mask = tokens.eq(self.pad_token_id) #被盖住的为1，否则为0
            dict = self.decoder(input_ids = tokens,encoder_hidden_states = encoder_output,
                                encoder_padding_mask = encoder_pad_mask,decoder_padding_mask = decoder_pad_mask,
                                decoder_causal_mask = self.causal_masks[:tokens.size(1), :tokens.size(1)], #取一部分
                                return_dict = True)

        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        hidden_state = self.dropout_layer(hidden_state)

        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1),
             self.src_start_index + src_tokens.size(-1)),
            fill_value=-1e24) # batch seq_len label_len(label是index))

        # +4 是因为有4个token
        tag_scores = F.linear(hidden_state,self.dropout_layer(
            self.decoder.embed_tokens.weight[self.label_start_id:self.label_start_id + 4]))

        # 3是因为前面有3个无关token
        logits[:,:,3:self.src_start_index] = tag_scores

        return logits

    def decode(self,tokens,state):
        return self.forward(tokens,state)[:-1] # 为啥又冒号负一？









