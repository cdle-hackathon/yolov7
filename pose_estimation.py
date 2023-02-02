import tqdm
import cv2
import torch
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def process_keypoints(video_path, model_path, output_path, extract=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load(model_path)
    model = weigths["model"]
    model = model.half().to(device)
    model.eval()
    video = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc("m", "p", "4", "v"), 30, (768, 1280))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm.tqdm(total=num_frames, desc="inf")
    while video.isOpened():
        ret, frame = video.read()
        if frame is None:
            break
        pbar.update(1)

        frame = letterbox(frame, 1280, stride=64, auto=True)[0]
        h, w = frame.shape[:2]
        frame = transforms.ToTensor()(frame)
        frame = torch.tensor(np.array([frame.numpy()]))
        frame = frame.to(device)
        frame = frame.half()
        output, _ = model(frame)
        with torch.set_grad_enabled(False):
            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml["nc"], nkpt=model.yaml["nkpt"], kpt_label=True)
            output = output_to_keypoint(output)

        if extract is False:
            nimg = frame[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        else:
            nimg = np.ones((h, w, 3), dtype=np.uint8) * 255

        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

        writer.write(nimg)
        torch.cuda.empty_cache()

    video.release()
    writer.release()


def main():
    process_keypoints(
        "inference/videos/demo.mp4",
        "weights/yolov7-w6-pose.pt",
        "inference/videos/output_pose.mp4",
        extract=True
    )


if __name__ == "__main__":
    main()
