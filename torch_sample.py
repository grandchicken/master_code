import torch

# torch.gather(input,dim,index) 等价于 input.gather(dim,index) 给定下标index，返回input在新的下标下重新排序的结果
# tokens.filp(dims = [1]) 花活，沿维度1进行翻转

# torch ne 按元素级比较两个tensor是否不相等，返回相同维度的tensor
# >>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
# tensor([[False, True], [True, False]])

# torch where() 函数作用是按照一定的规则合并两个tensor类型
# torch.where(condition,a,b) 其中 输入参数condition是条件限制，如果满足条件选择a输出，否则选择b输出

# Tensor.new_full(size,fill_value)dd Returns a Tensor of size size filled with fill_value.

#  F.linear(input, weight) torch里面的linear函数

# new_zeros() 其可以方便的复制原来tensor的所有类型，比如数据类型和数据所在设备等等