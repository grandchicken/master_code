import torch

class BARTPredictor:

    def __init__(self,args,model,decoder,):
        self.args = args
        self.model = model
        self.decoder = decoder

    def predict(self,input_ids,attention_mask,mner_label,max_length = 20, max_len_a = 0.0, bos_token_id = None, eos_token_id  = None,
                repetition_penalty=1.0, length_penalty=1.0, pad_token_id=0, restricter=None):
        state = self.model.prepare_state(input_ids,attention_mask)
        tgt_tokens = mner_label.to(input_ids.device)
        result = no_beam_search_generate(self.args,decoder=self.decoder,state=state,tokens=tgt_tokens,max_length= max_length, max_len_a = max_len_a, bos_token_id = bos_token_id,
                                         eos_token_id = eos_token_id, repetition_penalty = repetition_penalty, length_penalty = length_penalty, pad_token_id = pad_token_id, restricter = restricter)
        return result


@torch.no_grad()
def no_beam_search_generate(args, decoder, state, tokens=None, max_length=20, max_len_a=0.0,
                             bos_token_id=None, eos_token_id=None, repetition_penalty=1.0, length_penalty=1.0,
                             pad_token_id=0, restricter=None):
    """
    贪婪地搜索句子
    :param Decoder decoder: Decoder对象
    :param torch.LongTensor tokens: batch_size x len, decode的输入值，如果为None，则自动从bos_token_id开始生成
    :param State state: 应该包含encoder的一些输出。
    :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
    :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
    :param int bos_token_id: 如果tokens传入为None，则使用bos_token_id开始往后解码。
    :param int eos_token_id: 结束的token，如果为None，则一定会解码到max_length这么长。
    :param int pad_token_id: pad的token id
    :param float repetition_penalty: 对重复出现的token多大的惩罚。
    :param float length_penalty: 对每个token（除了eos）按照长度进行一定的惩罚。
    :return:
    """

    device = args.device
    # 若tokens为None 那么只填一个bos
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError(
                "You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.num_samples
        if batch_size is None:
            raise RuntimeError(
                "Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1],
                            fill_value=bos_token_id,
                            dtype=torch.long).to(device)
    batch_size = tokens.size(0)

    if state.num_samples:
        assert state.num_samples == batch_size, "The number of samples in `tokens` and `state` should match."

    # 若eos不存在，规定为-1
    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    scores = decoder.decode(tokens=tokens, state=state)  # 主要是为了update state
    # 这里需要考虑如果在第一个位置就结束的情况
    # if _eos_token_id!=-1:
    #     scores[:, _eos_token_id] = -1e12

    if restricter is not None:
        _, next_tokens = restricter(state, tokens, scores, num_beams=1)
    else:
        # 解下一个token是什么
        next_tokens = scores.argmax(dim=-1, keepdim=True)
    # 解出来的拼在原token上
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1) # 当前的长度
    dones = token_ids.new_zeros(batch_size).eq(1).__or__(
        next_tokens.squeeze(1).eq(eos_token_id)) #__or__ 按位或运算
    # tokens = tokens[:, -1:]

    if max_len_a != 0:
        # (bsz x num_beams, )
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float() *
                           max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0), ),
                                          fill_value=max_length,
                                          dtype=torch.long)
        real_max_length = max_lengths.max().item() # 取该batch中最大的length
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(
                state.encoder_mask.size(0)).long() * max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0), ),
                                          fill_value=max_length,
                                          dtype=torch.long)

    # 未达到长度限制的情况下
    while cur_len < real_max_length:
        scores = decoder.decode(tokens=token_ids,
                                state=state)  # batch_size x vocab_size
        # 重复出现token的惩罚,为啥这样就惩罚了？
        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()  # 为什么score会小于0?
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        # 如果存在终止符且存在长度惩罚
        if eos_token_id is not None and length_penalty != 1.0:
            token_scores = scores / cur_len ** length_penalty  # batch_size x vocab_size
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)
            scores = scores.masked_scatter(
                eos_mask, token_scores)  # 也即除了eos，其他词的分数经过了放大/缩小

        if restricter is not None:
            _, next_tokens = restricter(state, token_ids, scores, 1)
        else:
            # 解下一个token
            next_tokens = scores.argmax(dim=-1, keepdim=True)
        next_tokens = next_tokens.squeeze(-1)

        # 如果已经达到对应的sequence长度了，就直接填为eos了
        if _eos_token_id != -1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len + 1),
                                                  _eos_token_id)
        next_tokens = next_tokens.masked_fill(
            dones, pad_token_id)  # 对已经搜索完成的sample做padding

        # 拼接已经解码出来的token
        tokens = next_tokens.unsqueeze(1)
        token_ids = torch.cat([token_ids, tokens],
                              dim=-1)  # batch_size x max_len

        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1

        # 如果都完成了
        if dones.min() == 1:
            break

    return token_ids