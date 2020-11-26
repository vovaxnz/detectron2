import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()

cfg = get_cfg()
print(cfg)
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = 'model_final_keypoint_rcnn_R_50_FPN_3x.pkl'
predictor = DefaultPredictor(cfg)

source_image_folder = 'images'
ann_folder = 'keypoints'


for i, image_name in enumerate(os.listdir(source_image_folder)):
    img_path = os.path.join(source_image_folder, image_name)
    img = cv2.imread(img_path)
    outputs = predictor(img)

    predictions = outputs['instances'].to('cpu')
    keypoints = predictions.pred_keypoints.numpy().tolist() if predictions.has("pred_keypoints") else None

    print(i)

    with open(os.path.join(ann_folder, f'{os.path.splitext(image_name)[0]}.json'), 'w')as jfile:
        json.dump(keypoints, jfile)