import yaml
import os

import cv2
import torch
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel


def convert_image(img_cv2, img_size=640, stride=32):
    # Padded resize
    img = letterbox(img_cv2, img_size, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img


def detect(config):
    save_dir = os.path.dirname(config["save_path"])
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(config["device"])
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(config["weight_path"], map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    # if config["trace"]:
    #     model = TracedModel(model, device, config["img_size"])

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(torch.zeros(1, 3, config["img_size"], config["img_size"]).to(device).type_as(next(model.parameters())))  # run once

    video = cv2.VideoCapture(config["source_path"])
    # writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc("m", "p", "4", "v"), 30, (768, 1280))
    while video.isOpened():
        ret, frame = video.read()
        if frame is None or not ret:
            break
        img = convert_image(frame, config["img_size"], stride)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, config["conf_thres"], config["iou_thres"], agnostic=config["agnostic_nms"])

        # Process detections
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=1)

            # Stream results
            if config["display"]:
                cv2.imshow("frame", frame)
                cv2.waitKey(1)


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    with torch.no_grad():
        detect(config)
