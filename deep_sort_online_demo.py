from __future__ import division, print_function, absolute_import
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools.generate_detections import create_box_encoder, generate_detections
from tools.generate_detections import ImageEncoder
from application_util.image_viewer import ImageViewer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument(
        "--images",
        dest='images',
        help="Image / Directory containing images to perform detection upon",
        default="mot_benchmark/test/KITTI-16/img1",
        type=str)
    parser.add_argument(
        "--confidence",
        dest="confidence",
        help="Object Confidence to filter predictions",
        default=0.5)
    parser.add_argument(
        "--detection_height",
        help="Threshold on the detection bounding box height. "
        "Detections with height smaller than this value are disregarded",
        default=0,
        type=int)
    parser.add_argument(
        "--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument(
        "--cfg",
        dest='cfgfile',
        help="Config file",
        default="cfg/yolov3.cfg",
        type=str)
    parser.add_argument(
        "--weights",
        dest='weightsfile',
        help="weightsfile",
        default="yolov3.weights",
        type=str)
    parser.add_argument(
        "--max_cosine_distance",
        help="Gating threshold for cosine distance "
        "metric (object appearance).",
        type=float,
        default=0.2)
    parser.add_argument(
        "--nn_budget",
        help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.",
        type=int,
        default=None)
    parser.add_argument(
        "--display",
        help="Show intermediate tracking results",
        default=True,
        type=bool)
    parser.add_argument(
        "--output_file",
        help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="tmp/hypotheses.txt")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    return parser.parse_args()


def create_detections(detection_mat, frame_idx, min_height=0):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[1:5], row[5], row[8:]
        bbox[2:] = bbox[2:] - bbox[:2]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(img, det, viewer, min_confidence, nms_max_overlap, tracker, display):
    def frame_callback(vis, frame_idx):
        print("Processing frame {:05d}".format(int(frame_idx) + 1))

        detections = create_detections(det, frame_idx)
        detections = [d for d in detections if d.confidence >= min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap,
                                                    scores)
        detections = [detections[i] for i in indices]
        tracker.predict()
        tracker.update(detections)

        if display:
            vis.set_image(img.copy())
            vis.draw_trackers(tracker.tracks)

    min_frame_idx = int(det[:, 0].min())
    max_frame_idx = int(det[:, 0].max())
    seq_info = {
        "image_size": img.shape[:2],
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx
    }
    if display:
        visualizer = visualization.Visualization(seq_info, viewer)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback, vedio_file='deep_sort1.avi')


args = arg_parse()
batch_size = 1
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
CUDA = torch.cuda.is_available()

num_classes = 20
classes = load_classes("data/voc.names")

load_model = time.time()
print("Loading model.....")
image_encoder = ImageEncoder(args.model)
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", args.max_cosine_distance, args.nn_budget)
tracker = Tracker(metric)
print("Model successfully loaded")
model.net_info["height"] = 416
inp_dim = 416
if CUDA:
    model.cuda()
model.eval()

read_dir = time.time()

try:
    imlist = [
        osp.join(osp.realpath('.'), args.images, img)
        for img in os.listdir(args.images)
    ]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), args.images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(args.images))
    exit()
imlist.sort()
load_batch = time.time()

init = 0
for i, img_path in enumerate(imlist):
    start_det_loop = time.time()

    start = time.time()
    img = cv2.imread(img_path)
    im_prep = prep_image(img, inp_dim)
    im_dim = [img.shape[:2][::-1]]
    if not init:
        viewer = ImageViewer(40, img.shape[:2][::-1], "Figure KITTI-16")
        init = 1
    im_dim = torch.FloatTensor(im_dim)
    if CUDA:
        im_prep = im_prep.cuda()
        im_dim = im_dim.cuda()
    with torch.no_grad():
        prediction = model(Variable(im_prep), CUDA)
    prediction = write_results(
        prediction, confidence, num_classes, nms_conf=nms_thesh)
    end = time.time()
    if isinstance(prediction, int):
        print("{:20s} predicted in {:6.3f} seconds".format(
            img_path.split("/")[-1], (end - start)))
        print("Objects Detected:")
        print("----------------------------------------------------------")
        continue
    prediction[:, 0] += i
    print("{:20s} predicted in {:6.3f} seconds".format(
        img_path.split("/")[-1], (end - start)))
    objs = [classes[int(x[-1])] for x in prediction]
    print("{:20s} {:s}".format("Objects Detected:", " ".join(objs)))
    print("----------------------------------------------------------")

    scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)
    prediction[:, [1, 3]] -= (
        inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
    prediction[:, [2, 4]] -= (
        inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
    prediction[:, 1:5] /= scaling_factor
    for j in range(prediction.shape[0]):
        prediction[j, [1, 3]] = torch.clamp(prediction[j, [1, 3]], 0.0,
                                            im_dim[0, 0])
        prediction[j, [2, 4]] = torch.clamp(prediction[j, [2, 4]], 0.0,
                                            im_dim[0, 1])
    if CUDA:
        torch.cuda.synchronize()
    output = prediction.cpu().numpy()
    encoder = create_box_encoder(image_encoder, batch_size=1)
    output = generate_detections(encoder, args.images, output)
    run(img, output, viewer, confidence, nms_thesh, tracker, args.display)

end_time = time.time()

print("{:25s}: {:03.04f}".format("load mode time", read_dir - load_model))
print("{:25s}: {:03.04f}".format("read dir time", load_batch - read_dir))
print("{:25s}: {:03.04f}".format("det-track time", end_time - load_batch))
print("{:25s}: {:03.04f}".format("aver det-track time",
                                 (end_time - load_batch) / len(imlist)))
