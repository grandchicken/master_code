import os
import io

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import numpy as np
import cv2
import torch
import PIL.Image

from tqdm import tqdm
import os
import pickle as pkl
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

# 初始化配置
config_path = 'config/faster_rcnn_r_101_C4_3x.yaml'
model_path = 'config/faster_rcnn_r_101_C4_3x.pkl'
cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

predictor = DefaultPredictor(cfg)

# 填入你需要处理图片所在文件夹
image_root = 'sample_image'
# faster-rcnn 抽取的region特征存储路径
output_att = 'output/_att'
# faster-rcnn 抽取的region box存储路径
output_box = 'output/_box'
NUM_OBJECTS = 36

image_stack = []
for root, dirs, files in os.walk(image_root):
    for name in files:
        image_stack.append((name,os.path.join(root,name)))


# 主处理流程
for img_name,img_path in tqdm(image_stack):
    img_id = img_name.split('.')[0]
    im = cv2.imread(img_path)

    with torch.no_grad():
        height,width = im.shape[:2]
        image = predictor.aug.get_transform(im).apply_image(im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = [{"image": image, "height": height, "width": width}]
        images = predictor.model.preprocess_image(inputs)
        features = predictor.model.backbone(images.tensor)
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]

        proposal_boxes_tensor = proposal.proposal_boxes.tensor[:NUM_OBJECTS].cpu().numpy()

        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )

        print(proposal_boxes_tensor.shape)


        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        feature_pooled_final = feature_pooled[:NUM_OBJECTS].cpu().numpy()
        print(feature_pooled_final.shape)
        np.save(os.path.join(output_att,img_id + '.npy'),feature_pooled_final) #36 2048
        np.save(os.path.join(output_box,img_id + '.npy'), proposal_boxes_tensor)