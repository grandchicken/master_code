import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# restnet_log_file = logging.FileHandler(filename='resnet_utils.log', mode='a', encoding='utf-8')
# fmt = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
# restnet_log_file.setFormatter(fmt)


# logger = logging.Logger(name = __name__,level=logging.INFO)
# logger.addHandler(restnet_log_file)
# logger.info('*********************************begin***********************************'
#             )
class myResnet(nn.Module):
    def __init__(self, resnet, if_fine_tune, device):
        super(myResnet, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
        self.device = device

    def forward(self, x, att_size=7):
        # x : [1 3 224 224]
        x = self.resnet.conv1(x) # 1 64 112 112 (224+6+1-7)/2 = 112
        x = self.resnet.bn1(x)  # 1 64 112 112
        x = self.resnet.relu(x) # 1 64 112 112
        x = self.resnet.maxpool(x) # 1 64 56 56

        x = self.resnet.layer1(x) # 1 256 56 56
        x = self.resnet.layer2(x) # 1 512 28 28
        x = self.resnet.layer3(x) # 1 1024 14 14
        x = self.resnet.layer4(x) # [1 2048 7 7]


        fc = x.mean(3).mean(2) # #1 2048
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]) #1 2048 7 7 (就算上面的x是1 2048 38 29,这里也依然是1 2048 7 7)
        # 因此输入图片只要不小于224*224即可

        x = self.resnet.avgpool(x) #1 2048 1 1
        x = x.view(x.size(0), -1) # 1 2048

        if not self.if_fine_tune:
            x= Variable(x.data)
            fc = Variable(fc.data)
            att = Variable(att.data)

        return x, fc, att


