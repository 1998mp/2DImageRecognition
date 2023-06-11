import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import pandas as pd
import numpy as np
import os, json, cv2, random
import glob
import IPython
from IPython import display
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


def imshow(img):
    _, ret = cv2.imencode('.jpg', img)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


def random_sample(dataset_dicts, number, metadata):
    for d in random.sample(dataset_dicts, number):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        imshow(vis.get_image()[:, :, ::-1])


def train(dataset_name,
          images_per_batch=2,
          learning_rate=0.00025,
          iteration=300,
          batch_size_per_image=512,
          num_classes=1,
          test=False,
          test_dataset_name="",
          init_model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
          retinaNet=False):
    cfg = get_cfg()

    # minimum image size for the train set
    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    # maximum image size for the train set
    cfg.INPUT.MAX_SIZE_TRAIN = 512
    # minimum image size for the test set
    cfg.INPUT.MIN_SIZE_TEST = 512
    # maximum image size for the test set
    cfg.INPUT.MAX_SIZE_TEST = 512

    # cfg.MODEL.DEVICE='cpu'

    cfg.merge_from_file(model_zoo.get_config_file(init_model))
    cfg.DATASETS.TRAIN = (dataset_name,)

    if test:
        cfg.DATASETS.TEST = (test_dataset_name,)
    else:
        cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(init_model)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = images_per_batch  # This is the real "batch size" commonly known to deep learning people (number of images per batch)
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = iteration
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image  # The "RoIHead batch size". 128 is faster, (default: 512). Number of regions per image used to train RPN.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    if retinaNet:
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    if test:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator(test_dataset_name, output_dir="./output")
        test_result = trainer.test(cfg, predictor.model, evaluator)
        return cfg, test_result

    return cfg, ""


def evaluate(image, metadata, predictor, instance_mode=False):
    im = cv2.imread(image)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    v = {}
    if instance_mode:
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
    else:
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.5,
                       )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # imshow(out.get_image()[:, :, ::-1])
    cv2.imshow("Result", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def random_test_sample(dataset_dicts, number, metadata, predictor, instance_mode=False):
    for i in range(number):
        images = glob.glob(dataset_dicts + "*.jpg")
        random_image = random.choice(images)
        evaluate(random_image, metadata, predictor, instance_mode)


def get_best_result(df, number=3, segmentation=True):
    dfbbox = df.loc[df['Result-type'] == "bbox"].sort_values(by='AP', ascending=False).head(number)
    if segmentation:
        dfsegm = df.loc[df['Result-type'] == "segm"].sort_values(by='AP', ascending=False).head(number)
        return dfbbox, dfsegm
    else:
        return dfbbox


def save_result(result, model_name, segmentation=True):
    df = pd.DataFrame()
    for i in range(len(result)):
        temp = pd.DataFrame(result[i][4], columns=result[i][4].keys())
        temp = temp.T

        if segmentation:
            images_per_batch = pd.Series([result[i][0], result[i][0]], index=[0, 1])
            learning_rate = pd.Series([result[i][1], result[i][1]], index=[0, 1])
            batch_size_per_image = pd.Series([result[i][2], result[i][2]], index=[0, 1])
        else:
            images_per_batch = pd.Series([result[i][0]], index=[0])
            learning_rate = pd.Series([result[i][1]], index=[0])
            batch_size_per_image = pd.Series([result[i][2]], index=[0])

        temp['images_per_batch'] = images_per_batch.values
        temp['learning_rate'] = learning_rate.values
        temp['batch_size_per_image'] = batch_size_per_image.values

        df = pd.concat([df, temp])
        df.to_csv('result/' + str(model_name) + '_temp.csv', index=True, index_label="Result-type")
