# YOLOv5/FFCA-YOLO local camera inference
"""
Run FFCA-YOLO inference on a local USB or built-in camera.

This script intentionally reuses this repository's detect.py inference stack:
DetectMultiBackend, letterbox, non_max_suppression, scale_boxes, Annotator.
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import LOGGER, check_img_size, cv2, non_max_suppression, print_args, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode, time_sync


WINDOW_NAME = 'FFCA-YOLO Local Camera'


class CameraReceiver:
    """Continuously reads camera frames and keeps only the latest frame."""

    def __init__(self, camera_id=0, width=0, height=0):
        self.camera_id = int(camera_id)
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.cap = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread = None
        self._latest_frame = None
        self._frame_id = 0
        self.fps = 0.0

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f'[ERROR] Cannot open camera {self.camera_id}')

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._running.set()
        self._thread = threading.Thread(target=self._receive, name='camera-receiver', daemon=True)
        self._thread.start()
        return self

    def _receive(self):
        while self._running.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_frame = frame
                self._frame_id += 1

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._latest_frame is None:
                return None, self._frame_id
            return self._latest_frame.copy(), self._frame_id

    def release(self):
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def load_model(weights, imgsz, device_str='0', half=False, data=ROOT / 'data/coco128.yaml'):
    device = select_device(device_str)
    fp16 = bool(half and device.type == 'cuda')
    if half and not fp16:
        LOGGER.warning('WARNING: --half requested but CUDA is not available/selected; using FP32 inference.')

    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=fp16)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1, 3, *imgsz))
    return model, stride, names, pt, imgsz


def preprocess_frame(frame, imgsz, stride, auto, device, fp16):
    im = letterbox(frame, imgsz, stride=stride, auto=auto)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC BGR to CHW RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if fp16 else im.float()
    im /= 255.0
    if len(im.shape) == 3:
        im = im[None]
    return im


@smart_inference_mode()
def run_inference(model, im, conf_thres, iou_thres, max_det=1000, classes=None, agnostic_nms=False):
    t1 = time_sync()
    pred = model(im)
    t2 = time_sync()
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    t3 = time_sync()
    return pred[0], (t2 - t1) * 1e3, (t3 - t2) * 1e3


def _name_for_class(names, cls):
    cls = int(cls)
    if isinstance(names, dict):
        return names.get(cls, str(cls))
    if cls < len(names):
        return names[cls]
    return str(cls)


def draw_detections(frame, det, names, input_shape, fps, latency_ms, line_thickness=2):
    im0 = np.ascontiguousarray(frame.copy())
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

    if len(det):
        det[:, :4] = scale_boxes(input_shape, det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f'{_name_for_class(names, c)} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))

    im0 = annotator.result()
    status = (f'FPS: {fps:.1f}', f'Inference: {latency_ms:.1f} ms')
    x0, y0, line_h = 10, 28, 26
    max_w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for s in status)
    cv2.rectangle(im0, (x0 - 6, y0 - 22), (x0 + max_w + 8, y0 + line_h + 8), (0, 0, 0), -1)
    for i, text in enumerate(status):
        y = y0 + i * line_h
        cv2.putText(im0, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return im0


def open_video_writer(path, frame_shape, fps):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame_shape[:2]
    writer_fps = fps if fps and fps > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output), fourcc, writer_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'[ERROR] Cannot open VideoWriter {output}')
    return writer


def cleanup(camera, writer=None):
    if writer is not None:
        writer.release()
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()


def run(opt):
    model, stride, names, pt, imgsz = load_model(
        opt.weights,
        opt.imgsz,
        device_str=opt.device,
        half=opt.half,
        data=opt.data,
    )

    camera = None
    writer = None
    try:
        camera = CameraReceiver(opt.camera_id, opt.camera_width, opt.camera_height).start()
        if opt.view_img:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        last_frame_id = -1
        fps_ema = 0.0
        while True:
            frame, frame_id = camera.read()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = frame_id

            loop_t0 = time_sync()
            im = preprocess_frame(frame, imgsz, stride, auto=pt, device=model.device, fp16=model.fp16)
            det, latency_ms, _ = run_inference(
                model,
                im,
                opt.conf_thres,
                opt.iou_thres,
                max_det=opt.max_det,
                classes=opt.classes,
                agnostic_nms=opt.agnostic_nms,
            )
            loop_ms = max((time_sync() - loop_t0) * 1e3, 1e-6)
            current_fps = 1e3 / loop_ms
            fps_ema = current_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * current_fps)

            annotated = draw_detections(
                frame,
                det,
                names,
                im.shape[2:],
                fps_ema,
                latency_ms,
                line_thickness=opt.line_thickness,
            )

            if opt.save_video:
                if writer is None:
                    writer = open_video_writer(opt.output, annotated.shape, camera.fps)
                writer.write(annotated)

            if opt.view_img:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
    except KeyboardInterrupt:
        LOGGER.info('Interrupted by user.')
    finally:
        cleanup(camera, writer)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, required=True, help='model path(s), e.g. runs/train/exp/weights/best.pt')
    parser.add_argument('--camera-id', type=int, default=0, help='local camera index')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference on CUDA')
    parser.add_argument('--view-img', action='store_true', help='show OpenCV display window')
    parser.add_argument('--save-video', action='store_true', help='save annotated camera stream')
    parser.add_argument('--output', type=str, default='outputs/local_camera_detection.mp4', help='output video path')
    parser.add_argument('--camera-width', type=int, default=0, help='requested camera frame width')
    parser.add_argument('--camera-height', type=int, default=0, help='requested camera frame height')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path for class names fallback')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness in pixels')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    run(opt)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
