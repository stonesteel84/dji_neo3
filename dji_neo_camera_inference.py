# YOLOv5/FFCA-YOLO DJI Neo live video inference
"""
Run FFCA-YOLO object detection on the DJI Neo live camera view.

DJI Neo is not exposed through DJI Mobile SDK as a Python/PC video device.
For the controller-free Android + Windows setup, this program mirrors DJI Fly
over USB with scrcpy and captures that window for low-latency inference.  The
phone remains connected to the aircraft over Wi-Fi and remains the controller.

Three input modes are available:

* scrcpy (default): launch a read-only Android mirror over USB and capture its
  Windows client area.  This is the mode for DJI Neo Mobile App Control.
* listen: start FFmpeg as a single-client RTMP receiver.  This is retained for
  controller configurations in which DJI Fly exposes RTMP live streaming.
* pull: read a stream from an existing RTMP/RTSP server with OpenCV.

The detection path intentionally reuses this repository's YOLOv5 stack:
DetectMultiBackend, letterbox, non_max_suppression, scale_boxes, Annotator.
"""

import argparse
import ctypes
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlsplit

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


WINDOW_NAME = 'FFCA-YOLO DJI Neo'
DEFAULT_LISTEN_URL = 'rtmp://0.0.0.0:1935/live/neo'
DEFAULT_PULL_URL = 'rtmp://127.0.0.1:1935/live/neo'
DEFAULT_SCRCPY_TITLE = 'DJI Neo Phone (scrcpy)'
DEFAULT_DATA = ROOT / 'data/AITOD.yaml'


class LatestFrameStore:
    """Thread-safe storage for only the newest decoded BGR frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_id = 0
        self._last_frame_time = None

    @staticmethod
    def as_bgr_uint8(frame):
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

    def put(self, frame):
        frame = self.as_bgr_uint8(frame)
        if frame is None:
            return False
        with self._lock:
            self._latest_frame = frame.copy()
            self._frame_id += 1
            self._last_frame_time = time.monotonic()
        return True

    def get(self) -> Tuple[Optional[np.ndarray], int, Optional[float]]:
        with self._lock:
            if self._latest_frame is None:
                return None, self._frame_id, self._last_frame_time
            return self._latest_frame.copy(), self._frame_id, self._last_frame_time


class FFmpegRtmpReceiver:
    """Receive a DJI Fly RTMP push directly with FFmpeg and decode MJPEG frames."""

    def __init__(self, listen_url=DEFAULT_LISTEN_URL, ffmpeg_path='ffmpeg', reconnect_delay=1.0):
        self.listen_url = str(listen_url)
        self.ffmpeg_path = str(ffmpeg_path)
        self.reconnect_delay = max(float(reconnect_delay), 0.1)
        self.frames = LatestFrameStore()

        self._resolved_ffmpeg = None
        self._stop_event = threading.Event()
        self._thread = None
        self._process = None
        self._process_lock = threading.Lock()
        self._stderr_tail = deque(maxlen=12)
        self._last_error = None

    def check_dependencies(self):
        parsed = urlsplit(self.listen_url)
        if parsed.scheme.lower() != 'rtmp' or not parsed.hostname:
            raise RuntimeError(
                '[ERROR] --rtmp-listen-url must be an RTMP URL, for example '
                'rtmp://0.0.0.0:1935/live/neo'
            )

        explicit_path = Path(self.ffmpeg_path).expanduser()
        if explicit_path.is_file():
            self._resolved_ffmpeg = str(explicit_path.resolve())
        else:
            self._resolved_ffmpeg = shutil.which(self.ffmpeg_path)

        if not self._resolved_ffmpeg:
            raise RuntimeError(
                '[ERROR] FFmpeg was not found. Install ffmpeg and make it available on PATH, '
                'or pass --ffmpeg-path /absolute/path/to/ffmpeg.'
            )

    def start_stream(self):
        if self._resolved_ffmpeg is None:
            self.check_dependencies()
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._supervise, name='dji-neo-rtmp', daemon=True)
        self._thread.start()
        LOGGER.info(f'DJI Neo RTMP receiver listening at {self.listen_url}')
        LOGGER.info(f'Enter {self.publisher_url()} in DJI Fly, then start the livestream.')

    def _command(self):
        return [
            self._resolved_ffmpeg,
            '-hide_banner',
            '-loglevel',
            'warning',
            '-listen',
            '1',
            '-i',
            self.listen_url,
            '-map',
            '0:v:0',
            '-an',
            '-sn',
            '-dn',
            '-f',
            'image2pipe',
            '-vcodec',
            'mjpeg',
            '-pix_fmt',
            'yuvj420p',
            '-q:v',
            '3',
            'pipe:1',
        ]

    def _supervise(self):
        while not self._stop_event.is_set():
            stderr_thread = None
            try:
                creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                process = subprocess.Popen(
                    self._command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    creationflags=creationflags,
                )
                with self._process_lock:
                    self._process = process

                stderr_thread = threading.Thread(
                    target=self._drain_stderr,
                    args=(process.stderr,),
                    name='dji-neo-ffmpeg-stderr',
                    daemon=True,
                )
                stderr_thread.start()
                self._decode_mjpeg_stream(process.stdout)
                return_code = process.wait()
                if not self._stop_event.is_set():
                    self._last_error = f'FFmpeg exited with code {return_code}'
                    LOGGER.warning(
                        f'WARNING: {self._last_error}; restarting the DJI Neo RTMP listener '
                        f'in {self.reconnect_delay:.1f}s.'
                    )
            except Exception as exc:
                self._last_error = f'FFmpeg receiver failed: {exc}'
                if not self._stop_event.is_set():
                    LOGGER.warning(f'WARNING: {self._last_error}')
            finally:
                with self._process_lock:
                    self._process = None
                if stderr_thread is not None:
                    stderr_thread.join(timeout=0.5)

            self._stop_event.wait(self.reconnect_delay)

    def _decode_mjpeg_stream(self, pipe):
        data = bytearray()
        max_jpeg_bytes = 16 * 1024 * 1024

        while not self._stop_event.is_set():
            chunk = pipe.read(64 * 1024)
            if not chunk:
                break
            data.extend(chunk)

            while True:
                start = data.find(b'\xff\xd8')
                if start < 0:
                    if len(data) > 1:
                        del data[:-1]
                    break
                if start:
                    del data[:start]

                end = data.find(b'\xff\xd9', 2)
                if end < 0:
                    if len(data) > max_jpeg_bytes:
                        self._last_error = 'Discarded an invalid oversized MJPEG frame'
                        del data[:2]
                    break

                encoded = np.frombuffer(bytes(data[:end + 2]), dtype=np.uint8)
                del data[:end + 2]
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is not None:
                    self._last_error = None
                    self.frames.put(frame)

    def _drain_stderr(self, pipe):
        try:
            for raw_line in iter(pipe.readline, b''):
                line = raw_line.decode('utf-8', errors='replace').strip()
                if line:
                    self._stderr_tail.append(line)
        except (OSError, ValueError):
            pass

    def publisher_url(self):
        """Return the RTMP URL to enter in DJI Fly when a LAN IP can be found."""
        parsed = urlsplit(self.listen_url)
        host = parsed.hostname
        if host in ('0.0.0.0', '127.0.0.1', 'localhost'):
            host = discover_lan_ip()
        port = f':{parsed.port}' if parsed.port else ''
        path = parsed.path or '/live/neo'
        return f'rtmp://{host}{port}{path}'

    def get_latest_frame(self):
        return self.frames.get()

    def diagnostic(self):
        details = []
        if self._last_error:
            details.append(self._last_error)
        if self._stderr_tail:
            details.append(self._stderr_tail[-1])
        return ' | '.join(details) if details else 'FFmpeg is waiting for a DJI Fly publisher.'

    def stop(self):
        self._stop_event.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=4.0)


class OpenCVStreamReceiver:
    """Pull an RTMP/RTSP/video source and expose only its newest decoded frame."""

    def __init__(self, source=DEFAULT_PULL_URL, reconnect_delay=1.0, buffer_size=1):
        self.source = normalize_source(source)
        self.reconnect_delay = max(float(reconnect_delay), 0.1)
        self.buffer_size = max(int(buffer_size), 1)
        self.frames = LatestFrameStore()

        self._stop_event = threading.Event()
        self._thread = None
        self._capture = None
        self._capture_lock = threading.Lock()
        self._last_error = None

    def check_dependencies(self):
        if self.source is None or self.source == '':
            raise RuntimeError('[ERROR] --source is required when --input-mode pull is selected.')

    def start_stream(self):
        self.check_dependencies()
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, name='dji-neo-stream-pull', daemon=True)
        self._thread.start()
        LOGGER.info(f'Pulling DJI Neo video from {self.source}')

    def _open_capture(self):
        if isinstance(self.source, int):
            capture = cv2.VideoCapture(self.source)
        else:
            capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.source)

        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        return capture

    def _read_loop(self):
        while not self._stop_event.is_set():
            capture = self._open_capture()
            with self._capture_lock:
                self._capture = capture

            if not capture.isOpened():
                self._last_error = f'Cannot open video source {self.source}'
                capture.release()
                with self._capture_lock:
                    self._capture = None
                self._stop_event.wait(self.reconnect_delay)
                continue

            self._last_error = None
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self._last_error = f'Video source stopped producing frames: {self.source}'
                    break
                self.frames.put(frame)

            capture.release()
            with self._capture_lock:
                self._capture = None
            self._stop_event.wait(self.reconnect_delay)

    def get_latest_frame(self):
        return self.frames.get()

    def diagnostic(self):
        return self._last_error or f'Waiting for frames from {self.source}'

    def stop(self):
        self._stop_event.set()
        with self._capture_lock:
            capture = self._capture
        if capture is not None:
            capture.release()
        if self._thread is not None:
            self._thread.join(timeout=4.0)


class ScrcpyWindowReceiver:
    """Mirror Android with scrcpy, then capture its visible Windows client area."""

    class _Point(ctypes.Structure):
        _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]

    class _Rect(ctypes.Structure):
        _fields_ = [
            ('left', ctypes.c_long),
            ('top', ctypes.c_long),
            ('right', ctypes.c_long),
            ('bottom', ctypes.c_long),
        ]

    def __init__(
            self,
            scrcpy_path='scrcpy',
            window_title=DEFAULT_SCRCPY_TITLE,
            serial=None,
            max_fps=30,
            max_size=960,
            crop=None,
            launch=True,
            always_on_top=True,
    ):
        self.scrcpy_path = str(scrcpy_path)
        self.window_title = str(window_title)
        self.serial = str(serial) if serial else None
        self.max_fps = int(max_fps)
        self.max_size = int(max_size)
        self.launch = bool(launch)
        self.always_on_top = bool(always_on_top)
        self.frames = LatestFrameStore()

        self._crop_lock = threading.Lock()
        self._crop = self._validate_crop(crop)
        self._resolved_scrcpy = None
        self._stop_event = threading.Event()
        self._thread = None
        self._process = None
        self._process_lock = threading.Lock()
        self._output_tail = deque(maxlen=20)
        self._last_error = None

    @staticmethod
    def _validate_crop(crop):
        if crop is None:
            return None
        if len(crop) != 4:
            raise ValueError('scrcpy crop must contain x, y, width, and height')
        x, y, width, height = (int(value) for value in crop)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError('scrcpy crop requires x/y >= 0 and width/height > 0')
        return x, y, width, height

    def set_crop(self, crop):
        validated = self._validate_crop(crop)
        with self._crop_lock:
            self._crop = validated

    def get_crop(self):
        with self._crop_lock:
            return self._crop

    def check_dependencies(self):
        if not sys.platform.startswith('win'):
            raise RuntimeError('[ERROR] --input-mode scrcpy currently requires Windows.')

        try:
            import mss  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                '[ERROR] Python package mss is required for scrcpy window capture. '
                'Install it with: python -m pip install mss==10.2.0'
            ) from exc

        if self.launch:
            explicit_path = Path(self.scrcpy_path).expanduser()
            if explicit_path.is_file():
                self._resolved_scrcpy = str(explicit_path.resolve())
            else:
                self._resolved_scrcpy = shutil.which(self.scrcpy_path)

            if not self._resolved_scrcpy:
                raise RuntimeError(
                    '[ERROR] scrcpy was not found. On Windows install it with '
                    '`winget install --exact Genymobile.scrcpy`, reopen PowerShell, '
                    'or pass --scrcpy-path C:\\path\\to\\scrcpy.exe.'
                )

    def _command(self):
        command = [
            self._resolved_scrcpy,
            '--no-control',
            '--no-audio',
            f'--max-fps={self.max_fps}',
            f'--max-size={self.max_size}',
            '--video-codec=h264',
            f'--window-title={self.window_title}',
            '--window-x=0',
            '--window-y=0',
        ]
        if self.always_on_top:
            command.append('--always-on-top')
        if self.serial:
            command.extend(('--serial', self.serial))
        return command

    def start_stream(self):
        self.check_dependencies()
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        if self.launch:
            creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            try:
                process = subprocess.Popen(
                    self._command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    creationflags=creationflags,
                )
            except OSError as exc:
                raise RuntimeError(f'[ERROR] Could not start scrcpy: {exc}') from exc
            with self._process_lock:
                self._process = process
            threading.Thread(
                target=self._drain_process_output,
                args=(process.stdout,),
                name='dji-neo-scrcpy-output',
                daemon=True,
            ).start()
            LOGGER.info('Started scrcpy in read-only mode (--no-control); fly the Neo from the phone.')
        else:
            LOGGER.info(f'Attaching to an existing visible window titled: {self.window_title}')

        self._thread = threading.Thread(
            target=self._capture_loop,
            name='dji-neo-scrcpy-capture',
            daemon=True,
        )
        self._thread.start()

    def _drain_process_output(self, pipe):
        if pipe is None:
            return
        try:
            for raw_line in iter(pipe.readline, b''):
                line = raw_line.decode('utf-8', errors='replace').strip()
                if line:
                    self._output_tail.append(line)
        except (OSError, ValueError):
            pass

    @staticmethod
    def _configure_dpi_awareness(user32):
        try:
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except (AttributeError, OSError):
                pass

    def _configure_user32(self):
        user32 = ctypes.windll.user32
        self._configure_dpi_awareness(user32)
        user32.FindWindowW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.GetClientRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(self._Rect))
        user32.GetClientRect.restype = ctypes.c_bool
        user32.ClientToScreen.argtypes = (ctypes.c_void_p, ctypes.POINTER(self._Point))
        user32.ClientToScreen.restype = ctypes.c_bool
        user32.IsIconic.argtypes = (ctypes.c_void_p,)
        user32.IsIconic.restype = ctypes.c_bool
        return user32

    def _client_area(self, user32, hwnd):
        if user32.IsIconic(hwnd):
            raise RuntimeError('The scrcpy window is minimized; restore it so Windows can capture it.')

        rect = self._Rect()
        origin = self._Point(0, 0)
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            raise RuntimeError('GetClientRect failed for the scrcpy window.')
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise RuntimeError('ClientToScreen failed for the scrcpy window.')

        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            raise RuntimeError('The scrcpy window has an empty client area.')
        return {'left': int(origin.x), 'top': int(origin.y), 'width': width, 'height': height}

    def _apply_crop(self, frame):
        with self._crop_lock:
            crop = self._crop
        if crop is None:
            return frame

        x, y, width, height = crop
        frame_height, frame_width = frame.shape[:2]
        if x + width > frame_width or y + height > frame_height:
            raise RuntimeError(
                f'--scrcpy-crop {x} {y} {width} {height} exceeds current scrcpy '
                f'client size {frame_width}x{frame_height}.'
            )
        return np.ascontiguousarray(frame[y:y + height, x:x + width])

    def _capture_loop(self):
        import mss

        try:
            user32 = self._configure_user32()
            capture_interval = 1.0 / self.max_fps
            # MSS 10.2 exposes the class as the stable public API.  The package
            # still keeps the old mss() factory for compatibility, but using the
            # class avoids a deprecation warning on current Windows installs.
            with mss.MSS() as screen_capture:
                while not self._stop_event.is_set():
                    capture_started = time.monotonic()
                    with self._process_lock:
                        process = self._process
                    if process is not None and process.poll() is not None:
                        self._last_error = f'scrcpy exited with code {process.returncode}'
                        self._stop_event.wait(0.25)
                        continue

                    hwnd = user32.FindWindowW(None, self.window_title)
                    if not hwnd:
                        self._last_error = f'Waiting for scrcpy window titled "{self.window_title}"'
                        self._stop_event.wait(0.1)
                        continue

                    try:
                        monitor = self._client_area(user32, hwnd)
                        bgra = np.asarray(screen_capture.grab(monitor), dtype=np.uint8)
                        frame = np.ascontiguousarray(bgra[:, :, :3])
                        frame = self._apply_crop(frame)
                        self.frames.put(frame)
                        self._last_error = None
                    except (RuntimeError, OSError, ValueError) as exc:
                        self._last_error = str(exc)
                        self._stop_event.wait(0.1)
                    else:
                        remaining = capture_interval - (time.monotonic() - capture_started)
                        if remaining > 0:
                            self._stop_event.wait(remaining)
        except Exception as exc:
            self._last_error = f'scrcpy window capture failed: {exc}'

    def get_latest_frame(self):
        return self.frames.get()

    def diagnostic(self):
        details = []
        if self._last_error:
            details.append(self._last_error)
        if self._output_tail:
            details.append(self._output_tail[-1])
        if details:
            return ' | '.join(details)
        return 'Waiting for the visible scrcpy window to produce a frame.'

    def stop(self):
        self._stop_event.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=4.0)


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
            data=DEFAULT_DATA,
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

    # Use no_grad instead of inference_mode here: the bundled YOLOv5 NMS path
    # performs in-place confidence updates, which PyTorch 2.6+ forbids on
    # inference tensors once they leave the inference context.
    @torch.no_grad()
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
        self._draw_status(im0, fps, latency_ms, len(det))
        return im0

    def _name_for_class(self, cls):
        if isinstance(self.names, dict):
            return self.names.get(cls, str(cls))
        if cls < len(self.names):
            return self.names[cls]
        return str(cls)

    @staticmethod
    def _draw_status(frame, fps, latency_ms, detections):
        status = (f'FPS: {fps:.1f}', f'Inference: {latency_ms:.1f} ms', f'Detections: {detections}')
        # Scale the overlay to the captured crop.  A fixed 0.7 font made a
        # 249x241 phone crop look like a giant black box when the window was
        # enlarged by OpenCV.
        font_scale = max(0.35, min(0.7, frame.shape[1] / 900.0))
        thickness = 1 if font_scale < 0.5 else 2
        x0 = max(6, int(frame.shape[1] * 0.02))
        y0 = max(20, int(32 * font_scale / 0.7))
        line_h = max(18, int(34 * font_scale / 0.7))
        padding = max(4, int(8 * font_scale / 0.7))
        max_w = max(
            cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
            for s in status
        )
        cv2.rectangle(
            frame,
            (x0 - padding, y0 - int(22 * font_scale / 0.7)),
            (min(frame.shape[1] - 1, x0 + max_w + padding), y0 + line_h * len(status) + padding),
            (0, 0, 0),
            -1,
        )
        for i, text in enumerate(status):
            y = y0 + i * line_h
            cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)


def discover_lan_ip():
    """Best-effort LAN address discovery without sending application data."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('8.8.8.8', 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return '<PC_LAN_IP>'
    finally:
        sock.close()


def normalize_source(source) -> Union[int, str, None]:
    if source is None:
        return None
    value = str(source).strip()
    return int(value) if value.isdecimal() else value


def build_receiver(opt):
    if opt.input_mode == 'scrcpy':
        return ScrcpyWindowReceiver(
            scrcpy_path=opt.scrcpy_path,
            window_title=opt.scrcpy_window_title,
            serial=opt.scrcpy_serial,
            max_fps=opt.scrcpy_max_fps,
            max_size=opt.scrcpy_max_size,
            crop=opt.scrcpy_crop,
            launch=not opt.scrcpy_no_launch,
            always_on_top=not opt.scrcpy_allow_occlusion,
        )
    if opt.input_mode == 'listen':
        return FFmpegRtmpReceiver(
            listen_url=opt.rtmp_listen_url,
            ffmpeg_path=opt.ffmpeg_path,
            reconnect_delay=opt.reconnect_delay,
        )
    return OpenCVStreamReceiver(
        source=opt.source,
        reconnect_delay=opt.reconnect_delay,
        buffer_size=opt.stream_buffer_size,
    )


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
    receiver = build_receiver(opt)
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
        receiver.start_stream()

        if opt.view_img:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            if opt.input_mode == 'scrcpy':
                cv2.moveWindow(WINDOW_NAME, opt.result_window_x, opt.result_window_y)

        last_frame_id = -1
        fps_ema = 0.0
        no_frame_deadline = time.monotonic() + opt.frame_timeout
        crop_selected = not opt.select_crop

        while True:
            frame, frame_id, _ = receiver.get_latest_frame()
            if frame is None or frame_id == last_frame_id:
                if time.monotonic() > no_frame_deadline:
                    raise RuntimeError(
                        '[ERROR] No new frames were received from the DJI Neo input.\n'
                        f'{receiver.diagnostic()}\n'
                        'Check the selected input mode and its connection/setup instructions.'
                    )
                time.sleep(0.001)
                continue

            last_frame_id = frame_id
            no_frame_deadline = time.monotonic() + opt.frame_timeout

            if not crop_selected:
                LOGGER.info('Drag the DJI Fly camera ROI, then press Enter/Space. Press Esc to cancel.')
                roi = cv2.selectROI('Select DJI Fly camera area', frame, False, False)
                cv2.destroyWindow('Select DJI Fly camera area')
                x, y, width, height = (int(value) for value in roi)
                if width <= 0 or height <= 0:
                    raise RuntimeError('[ERROR] scrcpy crop selection was cancelled or empty.')
                receiver.set_crop((x, y, width, height))
                LOGGER.info(
                    'Selected camera ROI. Reuse it next time with: '
                    f'--scrcpy-crop {x} {y} {width} {height}'
                )
                crop_selected = True
                continue

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


def parse_opt(args=None):
    parser = argparse.ArgumentParser(
        description='Run FFCA-YOLO inference on DJI Neo video shown in DJI Fly.'
    )
    parser.add_argument('--weights', nargs='+', type=str, required=True, help='model path(s), e.g. ffca_yolo/weights/best.pt')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference on CUDA')
    parser.add_argument('--view-img', action='store_true', help='show OpenCV display window')
    parser.add_argument('--save-video', action='store_true', help='save annotated video')
    parser.add_argument('--output', type=str, default='outputs/dji_neo_detection.mp4', help='output video path')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA, help='dataset.yaml path for class names fallback')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness in pixels')

    stream = parser.add_argument_group('DJI Neo video input')
    stream.add_argument(
        '--input-mode',
        choices=('scrcpy', 'listen', 'pull'),
        default='scrcpy',
        help='scrcpy: Android USB mirror (default); listen: FFmpeg RTMP receiver; pull: existing stream',
    )
    stream.add_argument('--scrcpy-path', type=str, default='scrcpy', help='scrcpy executable name or path')
    stream.add_argument(
        '--scrcpy-window-title',
        type=str,
        default=DEFAULT_SCRCPY_TITLE,
        help='exact scrcpy window title to launch or attach to',
    )
    stream.add_argument('--scrcpy-serial', type=str, help='ADB device serial when more than one device is connected')
    stream.add_argument('--scrcpy-max-fps', type=int, default=30, help='Android mirror FPS limit')
    stream.add_argument('--scrcpy-max-size', type=int, default=960, help='maximum mirrored video dimension')
    stream.add_argument(
        '--scrcpy-crop',
        nargs=4,
        type=int,
        metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
        help='camera ROI relative to the scrcpy client area',
    )
    stream.add_argument(
        '--select-crop',
        action='store_true',
        help='interactively select the DJI Fly camera ROI from the first scrcpy frame',
    )
    stream.add_argument(
        '--scrcpy-no-launch',
        action='store_true',
        help='capture an already-running window instead of launching scrcpy',
    )
    stream.add_argument(
        '--scrcpy-allow-occlusion',
        action='store_true',
        help='do not keep scrcpy always-on-top (the window must still remain unobscured)',
    )
    stream.add_argument('--result-window-x', type=int, default=980, help='annotated result window X position')
    stream.add_argument('--result-window-y', type=int, default=0, help='annotated result window Y position')
    stream.add_argument(
        '--rtmp-listen-url',
        type=str,
        default=DEFAULT_LISTEN_URL,
        help='local RTMP address used in listen mode',
    )
    stream.add_argument(
        '--ffmpeg-path',
        type=str,
        default='ffmpeg',
        help='FFmpeg executable name or path used in listen mode',
    )
    stream.add_argument(
        '--source',
        type=str,
        default=DEFAULT_PULL_URL,
        help='RTMP/RTSP URL or capture device used in pull mode',
    )
    stream.add_argument('--stream-buffer-size', type=int, default=1, help='OpenCV capture buffer size in pull mode')
    stream.add_argument('--reconnect-delay', type=float, default=1.0, help='seconds before restarting a disconnected receiver')
    stream.add_argument('--frame-timeout', type=float, default=60.0, help='seconds to wait for the first/new stream frame')
    stream.add_argument('--output-fps', type=float, default=30.0, help='FPS metadata for saved annotated video')

    opt = parser.parse_args(args)
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    if len(opt.imgsz) != 2:
        parser.error('--imgsz accepts one value or exactly two values (height width).')
    if opt.frame_timeout <= 0:
        parser.error('--frame-timeout must be greater than zero.')
    if opt.scrcpy_max_fps <= 0:
        parser.error('--scrcpy-max-fps must be greater than zero.')
    if opt.scrcpy_max_size <= 0:
        parser.error('--scrcpy-max-size must be greater than zero.')
    if opt.select_crop and opt.input_mode != 'scrcpy':
        parser.error('--select-crop is only valid with --input-mode scrcpy.')
    if opt.select_crop and opt.scrcpy_crop:
        parser.error('--select-crop and --scrcpy-crop cannot be used together.')
    if opt.scrcpy_crop:
        try:
            ScrcpyWindowReceiver._validate_crop(opt.scrcpy_crop)
        except ValueError as exc:
            parser.error(str(exc))
    print_args(vars(opt))
    return opt


def main(opt):
    run(opt)


if __name__ == '__main__':
    main(parse_opt())
