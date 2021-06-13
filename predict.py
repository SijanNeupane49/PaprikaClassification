"""
Usage:
    # Apply predection to an image
    python3 predict.py <path to file>

    > default path is set to "./Paprika/train/1.jpg" if not path is given
"""

import torch
import cv2
import os
import sys

from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
# This allows running predictions without GPU
cfg.MODEL.DEVICE='cpu'

def visuallize_prediction(predictor, path):
    im = cv2.imread(path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1], scale=0.5,)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

predictor = DefaultPredictor(cfg)

default_predict_img_path = os.path.abspath("./Paprika/train/1.jpg")

predict_img = default_predict_img_path
if (len(sys.argv) >= 2):
    # ['predict.py', '/path/to/image']
    predict_img = sys.argv[1]

print("Predicting: " + predict_img)
visuallize_prediction(predictor, predict_img)
