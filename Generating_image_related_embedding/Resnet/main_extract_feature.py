from resnet_utils import myResnet
import resnet
import torch
from PIL import Image
from torchvision import transforms
import os

# 定义对图片的处理流程
transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# 处理图片

def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def extract_features(image_path,resnet_path):
    # 加载预训练模型
    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(resnet_path))
    # 定义模型
    model = myResnet(net,False,'cuda').to('cuda')
    image = image_process(image_path,transform).to('cuda')
    # print(image.shape)
    image = image.unsqueeze(0) # 1 3
    x,fc,att = model(image)
    # print(x.shape) #
    # print(fc.shape) # 1 2048
    # print(att.shape) # 1 2048 7 7 获得的特征是这个
    att = att.reshape((-1,2048)) #转换为 [49 2048]
    return att

resnet_path = '/home/data_ti6_d/lich/pretrain_model/resnet152.pth'
target_path = 'extract_features'
if not os.path.exists(target_path):
    os.mkdir(target_path)
for root,dirs,files in os.walk('sample_image'):
    for name in files:
        image_path = os.path.join(root,name)
        att = extract_features(image_path,resnet_path)
        print(att.shape)
        image_name = name.split('.')[0]
        torch.save(att,os.path.join(target_path,image_name+'.pt'))
