import numpy as np

## npz文件
path = 'DataResource/-homebrew-honey-beer-_1_2.npz'
b = np.load(path)
print(b['feat'].shape)