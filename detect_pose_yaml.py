import yaml
import os

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel


# def load():
#     # Padded resize
#     img = letterbox(img0, self.img_size, stride=self.stride)[0]

#     # Convert
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#     img = np.ascontiguousarray(img)

#     return path, img, img0, self.cap


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

    # Set Dataloader
    dataset = LoadImages(config["source_path"], img_size=config["img_size"], stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(torch.zeros(1, 3, config["img_size"], config["img_size"]).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = config["img_size"]
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != "cpu" and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, config["conf_thres"], config["iou_thres"], agnostic=config["agnostic_nms"])
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = "", im0s

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS")

            # Stream results
            if config["display"]:
                cv2.imshow("frame", im0)
                cv2.waitKey(1)


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    with torch.no_grad():
        detect(config)
