import torch, torchvision
import cv2

print(torch.__version__, torch.cuda.is_available())
print(cv2.__version__)

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
# For Tensorboard (Easy Graphs)
setup_logger()

import numpy as np
import os, json, cv2, random

import sys
INSIDE_COLAB = 'google.colab' in sys.modules

if INSIDE_COLAB:
    from google.colab.patches import cv2_imshow
    print("Inside Google Colab")
# cv2.imshow crashes in google colab hence using a google colab patch



# detectron2 imports
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode


"""
Call this function when outputting the image. Handles colab quirks for
outputting an image

"""
def handle_cv2imshow(img):
    if INSIDE_COLAB:
        cv2_imshow(img)
    else:
        cv2.imshow("AnnotatedImage", img)
        cv2.waitKey(0)



def get_paprika_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width


        annos = v["regions"]
        objs = []
        # for _, anno in annos.items():
        for anno in annos:
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                # Create a bounding box around the segmented pepper
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                # Extract Segmentation of where the pepper is
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("paprika_" + d, lambda d=d: get_paprika_dicts("Paprika/" + d))
    MetadataCatalog.get("paprika_" + d).set(thing_classes=["paprika"])

paprika_metadata = MetadataCatalog.get("paprika_train")

def visuallize_image(d):
    # 'd' === one annotation in the via file
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=paprika_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2imshow(out.get_image()[:, :, ::-1])


def train(dataset_name):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # Get initial configuration from model zoo
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0125  # pick a good LR # learing rate was 0.01,0.02, 0.00025
    cfg.SOLVER.MAX_ITER =1500   # 500,1000,300 was before, and 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (paprika). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return cfg

def update_configuration(cfg):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    return cfg


def visuallize_prediction(predictor, d):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=paprika_metadata,
                   scale=0.5,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])


dataset_dicts = get_paprika_dicts("Paprika/train")

for d in random.sample(dataset_dicts, 2):
    # View Some Annomtated Image Images to check if they have been annotated
    # properly
    visuallize_image(d)

# Train -> Update Configuration with Trained Model -> Get Ready for Prediction
cfg = train("paprika_train")
cfg = update_configuration(cfg)
predictor = DefaultPredictor(cfg)

dataset_dicts = get_paprika_dicts("Paprika/val")

for d in random.sample(dataset_dicts, 3):
    visuallize_prediction(predictor, d)


# Look at training curves in tensorboard:
# Only Enabled In Colab
# %load_ext tensorboard
# %tensorboard --logdir output
