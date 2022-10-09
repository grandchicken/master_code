import torch
import torch.nn as nn
import torch.nn.functional as F


def cmp(v1, v2):
	if v1[0] == v2[0]:
		return v1[1] - v2[1]
	return v1[0] - v2[0]

class Span_Loss(nn.Module):
	def __init__(self):
		super(Span_Loss, self).__init__()
	def forward(self,target,logits,mask):
		target = target.masked_fill(mask.eq(0),-100)
		output = F.cross_entropy(target,logits.transpose(1,2))
		return output


def liner_warmup(cur_step, t_step, warmup):
	# 这里是针对学习率的一种优化方式，见readme.md,在训练开始的时候先选择使用一个较小的学习率，
	# 训练了一些epoches,再修改为预先设置的学习率来进行训练
	progress = cur_step / t_step
	if progress < warmup:
		return progress / warmup
	return max((progress - 1.) / (warmup - 1.), 0.)

def set_lr(optimizer, lr):
	for group in optimizer.param_groups:
		group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
	for group in optimizer.param_groups:
		for param in group['params']:
			# print(param.shape)
			if param.grad == None:
				continue
			param.grad.data.clamp_(-grad_clip, grad_clip)