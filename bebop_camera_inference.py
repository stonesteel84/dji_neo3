# YOLOv5/FFCA-YOLO Parrot Bebop 2 video inference
"""
Run FFCA-YOLO object detection on a Parrot Bebop 2 video stream.

This script receives decoded frames through pyparrot DroneVision and reuses
this repository's detect.py inference stack: DetectMultiBackend, letterbox,
non_max_suppression, scale_boxes, Annotator.
"""

import argparse
import importlib.util
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


WINDOW_NAME = 'FFCA-YOLO Bebop 2'


class BebopVideoReceiver:
    """Bebop 2 video receiver that exposes only the latest decoded BGR frame."""

    def __init__(
            self,
            drone_type='Bebop2',
            ip_address=None,
            connect_retries=5,
            buffer_size=2,
            cleanup_old_images=True,
    ):
        self.drone_type = drone_type
        self.ip_address = ip_address
        self.connect_retries = int(connect_retries)
        self.buffer_size = max(int(buffer_size), 1)
        self.cleanup_old_images = bool(cleanup_old_images)

        self.Bebop = None
        self.DroneVision = None
        self.Model = None
        self.bebop = None
        self.vision = None
        self.connected = False
        self.stream_started = False

        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_id = 0
        self._last_frame_time = None

    def check_dependencies(self):
        if importlib.util.find_spec('pyparrot') is None:
            raise RuntimeError('[ERROR] pyparrot is not installed.')
        try:
            from pyparrot.Bebop import Bebop
            from pyparrot.DroneVision import DroneVision
            try:
                from pyparrot.Model import Model
            except ImportError:
                Model = None
        except ImportError as exc:
            raise RuntimeError(f'[ERROR] pyparrot import failed: {exc}') from exc

        self.Bebop = Bebop
        self.DroneVision = DroneVision
        self.Model = Model

    def connect(self):
        if self.Bebop is None:
            self.check_dependencies()

        try:
            try:
                self.bebop = self.Bebop(drone_type=self.drone_type, ip_address=self.ip_address)
            except TypeError:
                self.bebop = self.Bebop(drone_type=self.drone_type)
            success = self.bebop.connect(self.connect_retries)
        except Exception as exc:
            raise RuntimeError(
                '[ERROR] Cannot connect to Parrot Bebop 2.\n'
                'Connect the Ubuntu PC to the Bebop 2 Wi-Fi network first.\n'
                f'Original error: {exc}'
            ) from exc

        if not success:
            raise RuntimeError(
                '[ERROR] Cannot connect to Parrot Bebop 2.\n'
                'Connect the Ubuntu PC to the Bebop 2 Wi-Fi network first.'
            )

        self.connected = True

    def _create_vision(self):
        if self.Model is not None and hasattr(self.Model, 'BEBOP'):
            try:
                return self.DroneVision(
                    self.bebop,
                    self.Model.BEBOP,
                    buffer_size=self.buffer_size,
                    cleanup_old_images=self.cleanup_old_images,
                )
            except TypeError:
                pass

        return self.DroneVision(
            self.bebop,
            is_bebop=True,
            buffer_size=self.buffer_size,
            cleanup_old_images=self.cleanup_old_images,
        )

    def start_stream(self):
        if not self.connected:
            self.connect()

        try:
            self.vision = self._create_vision()
            self.vision.set_user_callback_function(self.frame_callback, user_callback_args=None)
            success = self.vision.open_video()
        except Exception as exc:
            raise RuntimeError(f'[ERROR] Cannot start Bebop 2 video stream.\nOriginal error: {exc}') from exc

        if not success:
            raise RuntimeError('[ERROR] Cannot start Bebop 2 video stream.')

        self.stream_started = True

    def frame_callback(self, _args):
        if self.vision is None:
            return

        frame = self.vision.get_latest_valid_picture()
        frame = self._as_bgr_uint8(frame)
        if frame is None:
            return

        with self._lock:
            self._latest_frame = frame.copy()
            self._frame_id += 1
            self._last_frame_time = time.monotonic()

    @staticmethod
    def _as_bgr_uint8(frame):
        if frame is None or not isinstance(frame, np.ndarray):
            return None

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            return None

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int, Optional[float]]:
        with self._lock:
            if self._latest_frame is None:
                return None, self._frame_id, self._last_frame_time
            return self._latest_frame.copy(), self._frame_id, self._last_frame_time

    def stop(self):
        if self.vision is not None and self.stream_started:
            try:
                self.vision.close_video()
            except Exception as exc:
                LOGGER.warning(f'WARNING: error while closing Bebop video stream: {exc}')
            finally:
                self.stream_started = False

        if self.bebop is not None and self.connected:
            try:
                self.bebop.disconnect()
            except Exception as exc:
                LOGGER.warning(f'WARNING: error while disconnecting Bebop 2: {exc}')
            finally:
                self.connected = False


class FFCAYoloDetector:
    """FFCA-YOLO detector using this repository's YOLOv5 inference utilities."""

    def __init__(
            self,
            weights,
            imgsz=(640, 640),
            conf_thres=0.25,
            iou_thres=0.45,
            device='0',
            half=False,
            data=ROOT / 'data/coco128.yaml',
            max_det=1000,
            classes=None,
            agnostic_nms=False,
            line_thickness=2,
    ):
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device_arg = device
        self.half = half
        self.data = data
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness

        self.model = None
        self.stride = None
        self.names = None
        self.pt = None

    def load_model(self):
        device = select_device(self.device_arg)
        fp16 = bool(self.half and device.type == 'cuda')
        if self.half and not fp16:
            LOGGER.warning('WARNING: --half requested but CUDA is not available/selected; using FP32 inference.')

        self.model = DetectMultiBackend(self.weights, device=device, dnn=False, data=self.data, fp16=fp16)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def preprocess(self, frame):
        im = letterbox(frame, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC BGR to CHW RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]
        return im

    @smart_inference_mode()
    def infer(self, frame):
        im = self.preprocess(frame)

        t1 = time_sync()
        pred = self.model(im)
        t2 = time_sync()
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            self.classes,
            self.agnostic_nms,
            max_det=self.max_det,
        )
        t3 = time_sync()
        return pred[0], im.shape[2:], (t2 - t1) * 1e3, (t3 - t2) * 1e3

    def draw(self, frame, det, input_shape, fps, latency_ms):
        im0 = np.ascontiguousarray(frame.copy())
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

        if len(det):
            det[:, :4] = scale_boxes(input_shape, det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{self._name_for_class(c)} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()
        self._draw_status(im0, fps, latency_ms)
        return im0

    def _name_for_class(self, cls):
        if isinstance(self.names, dict):
            return self.names.get(cls, str(cls))
        if cls < len(self.names):
            return self.names[cls]
        return str(cls)

    @staticmethod
    def _draw_status(frame, fps, latency_ms):
        status = (f'FPS: {fps:.1f}', f'Inference: {latency_ms:.1f} ms')
        x0, y0, line_h = 10, 28, 26
        max_w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for s in status)
        cv2.rectangle(frame, (x0 - 6, y0 - 22), (x0 + max_w + 8, y0 + line_h + 8), (0, 0, 0), -1)
        for i, text in enumerate(status):
            y = y0 + i * line_h
            cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def open_video_writer(path, frame_shape, fps=30.0):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output), fourcc, fps if fps > 0 else 30.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'[ERROR] Cannot open VideoWriter {output}')
    return writer


def cleanup(receiver, writer=None):
    if writer is not None:
        writer.release()
    if receiver is not None:
        receiver.stop()
    cv2.destroyAllWindows()


def run(opt):
    receiver = BebopVideoReceiver(
        drone_type='Bebop2',
        ip_address=opt.bebop_ip,
        connect_retries=opt.connect_retries,
        buffer_size=opt.bebop_buffer_size,
        cleanup_old_images=not opt.keep_pyparrot_images,
    )
    receiver.check_dependencies()

    detector = FFCAYoloDetector(
        weights=opt.weights,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.device,
        half=opt.half,
        data=opt.data,
        max_det=opt.max_det,
        classes=opt.classes,
        agnostic_nms=opt.agnostic_nms,
        line_thickness=opt.line_thickness,
    )
    detector.load_model()

    writer = None
    try:
        receiver.connect()
        receiver.start_stream()

        if opt.view_img:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        last_frame_id = -1
        fps_ema = 0.0
        no_frame_deadline = time.monotonic() + opt.frame_timeout

        while True:
            frame, frame_id, _ = receiver.get_latest_frame()
            if frame is None:
                if time.monotonic() > no_frame_deadline:
                    raise RuntimeError(
                        '[ERROR] No decoded frames received from Bebop 2 video stream.\n'
                        'Check Wi-Fi connection, ffmpeg, and pyparrot video stream setup.'
                    )
                time.sleep(0.001)
                continue

            no_frame_deadline = time.monotonic() + opt.frame_timeout
            if frame_id == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = frame_id

            loop_t0 = time_sync()
            det, input_shape, latency_ms, _ = detector.infer(frame)
            loop_ms = max((time_sync() - loop_t0) * 1e3, 1e-6)
            current_fps = 1e3 / loop_ms
            fps_ema = current_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * current_fps)

            annotated = detector.draw(frame, det, input_shape, fps_ema, latency_ms)

            if opt.save_video:
                if writer is None:
                    writer = open_video_writer(opt.output, annotated.shape, opt.output_fps)
                writer.write(annotated)

            if opt.view_img:
                cv2.imshow(WINDOW_NAME, annotated)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    except KeyboardInterrupt:
        LOGGER.info('Interrupted by user.')
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
    finally:
        cleanup(receiver, writer)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, required=True, help='model path(s), e.g. runs/train/exp/weights/best.pt')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference on CUDA')
    parser.add_argument('--view-img', action='store_true', help='show OpenCV display window')
    parser.add_argument('--save-video', action='store_true', help='save annotated video')
    parser.add_argument('--output', type=str, default='outputs/bebop_detection.mp4', help='output video path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path for class names fallback')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness in pixels')
    parser.add_argument('--bebop-ip', type=str, default=None, help='optional Bebop 2 IP address; pyparrot default is used if omitted')
    parser.add_argument('--connect-retries', type=int, default=5, help='pyparrot Bebop.connect retry count')
    parser.add_argument('--bebop-buffer-size', type=int, default=2, help='small pyparrot frame buffer; latest frame is used for inference')
    parser.add_argument('--frame-timeout', type=float, default=10.0, help='seconds to wait for decoded frames after stream start')
    parser.add_argument('--output-fps', type=float, default=30.0, help='FPS metadata for saved annotated video')
    parser.add_argument('--keep-pyparrot-images', action='store_true', help='do not delete old pyparrot ffmpeg image files on startup')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    run(opt)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
