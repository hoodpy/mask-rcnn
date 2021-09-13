import os
import numpy as np
import random
import math
import skimage
import matplotlib
import matplotlib.pyplot as plt
import model
import utils
import visualize
import coco_train


coco_model_path = "D:/program/mask_rcnn/checkpoint/mask_rcnn_coco.h5"
image_dir = "D:/program/mask_rcnn/demo/"
batch_size = 1
config = coco_train.CocoConfig()
network = model.MaskRCNN(batch_size, "inference", config)
network.load_weights(coco_model_path, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

for name in os.listdir(image_dir):
    image = skimage.io.imread(os.path.join(image_dir, name))
    result = network.detect(np.expand_dims(image, 0), verbose=1)[0]
    visualize.display_instances(image, result["rois"], result["masks"], result["class_ids"], class_names, result["scores"])