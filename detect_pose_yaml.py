import os

import cv2
import numpy as np
import torch
import tqdm
import yaml
from numpy import random
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, non_max_suppression_kpt, scale_coords, set_logging
from utils.plots import output_to_keypoint, plot_one_box, plot_skeleton_kpts
from utils.torch_utils import TracedModel


def load_pose(device, config):
    weigths = torch.load(config["pose_weight_path"])
    model = weigths["model"]
    model = model.half().to(device)
    model.eval()
    return model


def load_detect(device, config):
    # Initialize
    set_logging()
    half = device.type != "cpu"  # half precision only supported on CUDA
    model = attempt_load(config["detect_weight_path"], map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    if config["trace"]:
        model = TracedModel(model, device, config["img_size"])
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != "cpu":
        model(torch.zeros(1, 3, config["img_size"], config["img_size"])
              .to(device).type_as(next(model.parameters())))

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    return model, half, stride, names, colors


def estimate_pose(model, device, img):
    img = letterbox(img, 1280, stride=64, auto=True)[0]
    h, w = img.shape[:2]
    img = transforms.ToTensor()(img)
    img = torch.tensor(np.array([img.numpy()]))
    img = img.to(device)
    img = img.half()
    output, _ = model(img)
    with torch.set_grad_enabled(False):
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml["nc"], nkpt=model.yaml["nkpt"], kpt_label=True)
        output = output_to_keypoint(output)

    img_res = img[0].permute(1, 2, 0) * 255
    img_res = img_res.cpu().numpy().astype(np.uint8)
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
    skeleton = np.ones((h, w, 3), dtype=np.uint8) * 255
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(skeleton, output[idx, 7:].T, 3)
        plot_skeleton_kpts(img_res, output[idx, 7:].T, 3)
    return skeleton, img_res


def convert_image(img_cv2, img_size=640, stride=32):
    # Padded resize
    img = letterbox(img_cv2, img_size, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img


def detect_pose(model, device, img, config, half):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, config["conf_thres"], config["iou_thres"], agnostic=config["agnostic_nms"])
    return pred


def main(config):
    save_dir = os.path.dirname(config["save_path"])
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load models
    model_pose = load_pose(device, config)
    model_detect, half, stride, names, colors = load_detect(device, config)

    video = cv2.VideoCapture(config["source_path"])
    fps = video.get(cv2.CAP_PROP_FPS)
    w_dst = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_dst = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if config["save_path"] != "":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(config["save_path"], fourcc, fps, (w_dst, h_dst))

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm.tqdm(total=num_frames, desc="inf")
    while video.isOpened():
        ret, frame = video.read()
        if frame is None or not ret:
            break
        pbar.update(1)

        skeleton, img_res = estimate_pose(model_pose, device, frame)
        img = convert_image(skeleton, config["img_size"], stride)
        pred = detect_pose(model_detect, device, img, config, half)

        # Process detections
        for det in pred:
            if len(det) == 0:
                continue

            # Rescale boxes from img_size to frame size
            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], frame.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} {conf:.2f}"
                plot_one_box(xyxy, img_res, label=label, color=colors[int(cls)], line_thickness=2)

        # Resize to fit the original frame
        h_head, w_head = 0, 0
        if img_res.shape[0] > h_dst:
            h_head = (img_res.shape[0] - h_dst) // 2
        if img_res.shape[1] > w_dst:
            w_head = (img_res.shape[1] - w_dst) // 2
        img_res = img_res[h_head : h_head + h_dst, w_head : w_head + w_dst]

        # Stream results
        if config["display"]:
            cv2.imshow("frame", img_res)
            cv2.waitKey(1)

        if config["save_path"] != "":
            writer.write(img_res)
        torch.cuda.empty_cache()

    video.release()
    if config["save_path"] != "":
        writer.release()


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    with torch.no_grad():
        main(config)
