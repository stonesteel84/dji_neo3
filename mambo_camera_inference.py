"""Mambo FPV Wi-Fi preview and FFCA-YOLO tracking.

Connect Windows to the Mambo Wi-Fi first (Internet access is not required).
Preview: python mambo_camera_inference.py --preview-only
Track: python mambo_camera_inference.py --weights ffca_yolo/weights/best.pt
The default 'rtp' backend over UDP ignores the Mambo SDP SPS/PPS (which do not match
its stream) and needs ffmpeg; the drone's RTSP-over-TCP framing broke under load in tests.
This program receives video only; it does not send flight commands.
"""

import argparse
import ctypes
import faulthandler
from collections import deque
from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from mambo_stream_probe import START_CODE, RtspClient, StreamStats, find_ffmpeg

LOG = logging.getLogger('mambo')
WINDOW = 'Mambo FPV - FFCA-YOLO'
DEFAULT_URL = 'rtsp://192.168.99.1/media/stream2'


class MamboVideoReceiver:
    """Single decoder thread; retain only the latest frame and receipt time."""

    def __init__(self, url, transport='udp', open_timeout=5., read_timeout=3., reconnect_delay=1.):
        self.url = url
        self.transport = transport
        self.open_timeout = open_timeout
        self.read_timeout = read_timeout
        self.reconnect_delay = reconnect_delay
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._frame = None
        self._frame_id = 0
        self._received_at = None
        self._generation = 0
        self.error = None

    def start(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'rtsp_transport;{self.transport}'
        self._thread = threading.Thread(target=self._receive, name='mambo-video', daemon=True)
        self._thread.start()
        return self

    def _receive(self):
        try:
            while not self._stop.is_set():
                cap = cv2.VideoCapture()
                try:
                    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.open_timeout * 1000),
                              cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout * 1000)]
                    if not cap.open(self.url, cv2.CAP_FFMPEG, params):
                        LOG.warning('RTSP open failed. Check Mambo Wi-Fi; try the other --rtsp-transport.')
                    else:
                        with self._lock:
                            self._generation += 1
                        LOG.info('Mambo RTSP connected (%s).', self.transport)
                        while not self._stop.is_set():
                            ok, frame = cap.read()
                            if not ok or frame is None:
                                LOG.warning('Video interrupted; reconnecting.')
                                break
                            with self._lock:
                                self._frame = frame
                                self._frame_id += 1
                                self._received_at = time.monotonic()
                finally:
                    # Only the decoder thread owns/releases this capture.
                    cap.release()
                    with self._lock:
                        self._frame = None
                self._stop.wait(self.reconnect_delay)
        except Exception as exc:
            self.error = str(exc)
            LOG.error('Video receiver stopped: %s', exc)

    def read(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return frame, self._frame_id, self._received_at, self._generation

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.open_timeout, self.read_timeout) + 2.)
            if self._thread.is_alive():
                LOG.warning('Decoder is still exiting; backend did not honor its timeout.')


# libVLC's own prototype passes chroma as a writable char[5]; python-vlc declares it
# c_char_p (a read-only copy), so declare it as a raw pointer to be able to request RV32.
VideoFormatCb = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                                 ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint))
VideoCleanupCb = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class VlcVideoReceiver:
    """Decode with libVLC into a BGRA memory buffer, independent of OpenCV FFmpeg.

    A supervisor thread owns stall detection and player stop/release/recreate, so a
    slow libVLC stop never blocks the UI loop; read() only copies the latest frame.
    VLC re-renders the last picture about every 80 ms after input stops, so callback
    activity alone does not prove the stream is alive: byte-identical re-renders are
    not counted as new frames, and stream progress comes from VLC's displayed-picture
    counter (a genuinely static scene still advances it).
    """

    def __init__(self, url, transport='tcp', width=0, height=0, network_caching=300,
                 read_timeout=3., reconnect_delay=1., stats_interval=5., drop_late_frames=False):
        self.url = url
        self.drop_late_frames = drop_late_frames
        self.transport = transport
        self.out_width, self.out_height = width, height  # 0 keeps the decoded size
        self.width = self.height = 0
        # The format callback reports the padded decoder buffer (e.g. 640x368 for a
        # 640x360 stream) and VLC stretches into it; the visible size comes later.
        self._visible_size = None
        self.network_caching = network_caching
        self.read_timeout = read_timeout
        self.reconnect_delay = reconnect_delay
        self.stats_interval = stats_interval
        self._generation = 0
        self._attempt_started = None
        self._retry_at = None
        self._reconnect_count = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._pixels_lock = threading.Lock()
        self._pitch = 0
        self._pixel_storage = self._pixels = self._pixel_address = None
        if width and height:
            self._allocate(width, height)
        self._frame = None
        self._frame_id = 0
        self._received_at = None
        self._progress_at = None
        self._displayed = None
        self._repeats = 0
        self._instance = self._player = self._media = None
        self._dll_directory = None
        self._last_stats = None
        self._next_stats = 0.

    def _allocate(self, width, height):
        # libVLC video callbacks require 32-byte aligned planes. Keep the backing
        # allocation alive and align each scanline as well as the plane address.
        self.width, self.height = width, height
        self._pitch = (width * 4 + 31) & ~31
        pixel_bytes = self._pitch * height
        self._pixel_storage = (ctypes.c_ubyte * (pixel_bytes + 31))()
        self._pixel_address = (ctypes.addressof(self._pixel_storage) + 31) & ~31
        self._pixels = (ctypes.c_ubyte * pixel_bytes).from_address(self._pixel_address)

    def start(self):
        if os.name == 'nt':
            for root in (os.environ.get('ProgramFiles'), os.environ.get('ProgramFiles(x86)')):
                if root:
                    folder = Path(root) / 'VideoLAN' / 'VLC'
                    if (folder / 'libvlc.dll').is_file():
                        self._dll_directory = os.add_dll_directory(str(folder))
                        os.environ.setdefault('PYTHON_VLC_LIB_PATH', str(folder / 'libvlc.dll'))
                        os.environ.setdefault('PYTHON_VLC_MODULE_PATH', str(folder))
                        break
        try:
            import vlc
        except (ImportError, OSError) as exc:
            self.stop()
            raise RuntimeError('VLC backend requires matching 64-bit VLC desktop and python-vlc. '
                               'Install VLC, then run: python -m pip install python-vlc') from exc
        self._vlc = vlc

        @vlc.CallbackDecorators.VideoLockCb
        def lock(_opaque, planes):
            self._pixels_lock.acquire()
            planes[0] = self._pixel_address
            return None

        @vlc.CallbackDecorators.VideoUnlockCb
        def unlock(_opaque, _picture, _planes):
            self._pixels_lock.release()

        @vlc.CallbackDecorators.VideoDisplayCb
        def display(_opaque, _picture):
            try:
                with self._pixels_lock:
                    bgra = np.ctypeslib.as_array(self._pixels).reshape(self.height, self._pitch // 4, 4)
                    bgra = bgra[:, :self.width]
                    frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                visible = self._visible_size
                if not self.out_width and visible and visible != (frame.shape[1], frame.shape[0]):
                    frame = cv2.resize(frame, visible, interpolation=cv2.INTER_AREA)
                self._publish(frame)
            except Exception as exc:
                self.error = f'VLC frame callback failed: {exc}'

        @VideoFormatCb
        def setup(_opaque, chroma, width, height, pitches, lines):
            try:
                source = width[0], height[0]
                out_w, out_h = (self.out_width, self.out_height) if self.out_width else self._visible_size or source
                ctypes.memmove(chroma, b'RV32', 4)
                width[0], height[0] = out_w, out_h
                with self._pixels_lock:
                    self._allocate(out_w, out_h)
                pitches[0], lines[0] = self._pitch, out_h
                LOG.info('VLC video format: decoded %dx%d, output %dx%d RV32.', *source, out_w, out_h)
                return 1
            except Exception as exc:
                self.error = f'VLC format callback failed: {exc}'
                return 0

        @VideoCleanupCb
        def cleanup(_opaque):
            pass

        # Keep Python callback objects alive until the player has stopped.
        self._callbacks = (lock, unlock, display)
        self._format_callbacks = (setup, cleanup)
        self._set_format_callbacks = vlc.dll['libvlc_video_set_format_callbacks']
        self._set_format_callbacks.argtypes = (ctypes.c_void_p, VideoFormatCb, VideoCleanupCb)
        self._set_format_callbacks.restype = None
        try:
            args = ['--no-audio', '--no-video-title-show', '--avcodec-hw=none']
            if not self.drop_late_frames:
                # Mambo frames were judged late and dropped (stats: lost >> new frames);
                # show them instead of freezing on the last picture.
                args += ['--no-drop-late-frames', '--no-skip-frames', '--no-avcodec-hurry-up']
            self._instance = vlc.Instance(*args)
            if self._instance is None:
                raise RuntimeError('Cannot initialize libVLC; check VLC installation and Python architecture.')
            self._open_player()
            self._thread = threading.Thread(target=self._supervise, name='mambo-vlc', daemon=True)
            self._thread.start()
            LOG.info('VLC receiver started (%s); waiting for decoded frames.', self.transport)
            return self
        except Exception:
            self.stop()
            raise

    def _open_player(self):
        self._player = self._instance.media_player_new()
        self._media = self._instance.media_new(self.url)
        self._media.add_option(f':network-caching={self.network_caching}')
        if self.transport == 'tcp':
            self._media.add_option(':rtsp-tcp')
        self._player.set_media(self._media)
        self._player.video_set_callbacks(*self._callbacks, None)
        self._set_format_callbacks(self._player._as_parameter_, *self._format_callbacks)
        with self._lock:
            self._frame = None
            self._received_at = self._progress_at = None
            self._generation += 1
        self._displayed = None
        self._attempt_started = time.monotonic()
        self._last_stats = None
        if self._player.play() == -1:
            raise RuntimeError('VLC could not start the Mambo stream.')

    def _close_player(self):
        if self._player is not None:
            self._player.stop()
            self._player.release()
            self._player = None
        if self._media is not None:
            self._media.release()
            self._media = None
        with self._lock:
            self._frame = None
            self._received_at = self._progress_at = None

    def _supervise(self):
        try:
            while not self._stop.wait(.1):
                now = time.monotonic()
                self._poll_progress(now)
                self._check_connection(now)
                if self.stats_interval and now >= self._next_stats:
                    self._next_stats = now + self.stats_interval
                    self._log_stats()
        except Exception as exc:
            self.error = f'VLC receiver stopped: {exc}'

    def _poll_progress(self, now):
        if self._media is None:
            return
        stats = self._vlc.MediaStats()
        if self._media.get_stats(stats) and stats.displayed_pictures != self._displayed:
            if self._displayed is not None:
                with self._lock:
                    self._progress_at = now
            self._displayed = stats.displayed_pictures

    def _log_stats(self):
        """Separate bytes received, frames decoded and frames delivered to this program."""
        if self._media is None:
            return
        stats = self._vlc.MediaStats()
        if not self._media.get_stats(stats):
            return
        with self._lock:
            frame_id, repeats = self._frame_id, self._repeats
        # read_bytes stays 0 for RTSP (live555); demux_read_bytes counts received stream data.
        current = dict(read_kb=stats.demux_read_bytes / 1024., decoded=stats.decoded_video,
                       displayed=stats.displayed_pictures, lost=stats.lost_pictures,
                       corrupted=stats.demux_corrupted, discontinuity=stats.demux_discontinuity,
                       callback=frame_id, repeats=repeats)
        if self._last_stats is not None:
            delta = {key: current[key] - self._last_stats[key] for key in current}
            LOG.info('VLC stats/%gs session %d: demuxed %.0f kB, decoded %d, displayed %d, lost %d, '
                     'demux corrupted %d, discontinuity %d, new frames %d, repeated re-renders %d.',
                     self.stats_interval, self._generation, delta['read_kb'], delta['decoded'],
                     delta['displayed'], delta['lost'], delta['corrupted'], delta['discontinuity'],
                     delta['callback'], delta['repeats'])
        self._last_stats = current

    def _check_connection(self, now):
        if self._retry_at is not None:
            if now >= self._retry_at:
                self._retry_at = None
                try:
                    self._open_player()
                    LOG.info('VLC reconnect attempt %d started.', self._reconnect_count)
                except RuntimeError as exc:  # keep retrying; the preview window stays open
                    LOG.warning('VLC reconnect attempt %d failed: %s', self._reconnect_count, exc)
                    self._close_player()
                    self._retry_at = now + self.reconnect_delay
            return
        if self._player is None:
            return
        if not self.out_width:
            size = self._player.video_get_size(0)
            if size and size[0] > 0 and size[1] > 0 and tuple(size) != self._visible_size:
                self._visible_size = tuple(size)
                LOG.info('VLC visible video size: %dx%d.', *size)
        state = self._player.get_state()
        with self._lock:
            last_received = self._alive_at()
        # Allow startup/probing more time than an established stream stall.
        idle_limit = max(10., self.network_caching / 1000. + self.read_timeout) if last_received is None else self.read_timeout
        stale = now - (last_received if last_received is not None else self._attempt_started) > idle_limit
        if state in (self._vlc.State.Error, self._vlc.State.Ended, self._vlc.State.Stopped) or stale:
            self._reconnect_count += 1
            LOG.warning('VLC stream interrupted (state=%s, stalled=%s); reconnecting (%d).',
                        state, stale, self._reconnect_count)
            self._close_player()
            self._retry_at = time.monotonic() + self.reconnect_delay

    def _publish(self, frame):
        with self._lock:
            previous = self._frame
            if previous is not None and previous.shape == frame.shape and np.array_equal(previous, frame):
                self._repeats += 1  # re-render of the last picture, not a new frame
                return
            self._frame = frame
            self._frame_id += 1
            self._received_at = time.monotonic()

    def _alive_at(self):
        times = [t for t in (self._received_at, self._progress_at) if t is not None]
        return max(times) if times else None

    def read(self):
        """Return (latest frame, frame id, last stream progress time, session generation)."""
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return frame, self._frame_id, self._alive_at(), self._generation

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15.)
            if self._thread.is_alive():
                # Releasing libVLC under a thread still inside player.stop() could crash.
                LOG.warning('VLC supervisor is still stopping the player; skipping libVLC release.')
                return
        self._retry_at = None
        self._close_player()
        if self._instance is not None:
            self._instance.release()
            self._instance = None
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None


class InBandParameterFilter:
    """Annex-B writer that discards everything before the first in-band SPS.

    The Mambo SDP advertises SPS/PPS (Main profile, CABAC) that do not match the
    stream it sends (High profile, CAVLC). Decoders primed from the SDP misparse
    every slice ('cabac_init_idc overflow') until the next in-band SPS/PPS, so only
    the parameter sets carried inside the stream are passed to the decoder.
    """

    def __init__(self, sink):
        self.sink = sink
        self.started = False
        self.skipped = 0

    def write(self, data):
        nal = data[len(START_CODE):]
        if not self.started:
            if not nal or nal[0] & 0x1F != 7:
                self.skipped += 1
                return
            self.started = True
        self.sink(data)


class RtpVideoReceiver:
    """Own RTSP/RTP depacketizer feeding an FFmpeg decoder process.

    Unlike OpenCV/VLC this ignores the SDP parameter sets (see InBandParameterFilter)
    and keeps per-session RTP loss counters, so damaged input stays visible.
    """

    def __init__(self, url, transport='udp', width=640, height=360, read_timeout=3.,
                 reconnect_delay=1., stats_interval=5.):
        self.url, self.transport = url, transport
        self.width, self.height = width, height
        self.read_timeout = read_timeout
        self.reconnect_delay = reconnect_delay
        self.stats_interval = stats_interval
        self.error = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._frame = None
        self._frame_id = 0
        self._received_at = None
        self._generation = 0
        self._decode_errors = 0
        self._process = None
        self._ffmpeg = find_ffmpeg()

    def start(self):
        if not self._ffmpeg:
            raise RuntimeError('RTP backend needs ffmpeg (winget install Gyan.FFmpeg.Essentials).')
        self._thread = threading.Thread(target=self._receive, name='mambo-rtp', daemon=True)
        self._thread.start()
        return self

    def _start_decoder(self):
        # No '-fflags nobuffer': with the raw H.264 demuxer it emitted 2 of 272 frames in tests.
        command = [self._ffmpeg, '-hide_banner', '-loglevel', 'error', '-flags', 'low_delay', '-probesize', '32768', '-analyzeduration', '0', '-threads', '1',
                   '-f', 'h264', '-i', 'pipe:0', '-vf', f'scale={self.width}:{self.height}',
                   '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-flush_packets', '1', 'pipe:1']
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=self._read_frames, args=(process, self._generation), daemon=True).start()
        threading.Thread(target=self._read_errors, args=(process,), daemon=True).start()
        return process

    def _read_frames(self, process, generation):
        size = self.width * self.height * 3
        while True:
            data = bytearray()
            while len(data) < size:
                try:
                    chunk = process.stdout.read(size - len(data))
                except (OSError, ValueError):
                    return
                if not chunk:
                    return
                data += chunk
            frame = np.frombuffer(bytes(data), np.uint8).reshape(self.height, self.width, 3)
            with self._lock:
                if generation != self._generation:
                    return
                self._frame = frame
                self._frame_id += 1
                self._received_at = time.monotonic()

    def _read_errors(self, process):
        for line in process.stderr:
            if getattr(process, 'stopping', False):
                continue  # broken-pipe messages caused by our own shutdown
            with self._lock:
                self._decode_errors += 1
                count = self._decode_errors
            if count <= 5 or count % 100 == 0:
                LOG.warning('FFmpeg decode error #%d: %s', count, line.decode(errors='replace').strip())

    def _stop_decoder(self, process):
        if process is None:
            return
        process.stopping = True
        try:
            process.stdin.close()  # EOF lets FFmpeg exit; the reader threads then see EOF
        except OSError:
            pass
        try:
            process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _receive(self):
        while not self._stop.is_set():
            client = process = None
            with self._lock:
                self._generation += 1
                self._frame = self._received_at = None
                generation = self._generation
            try:
                client = RtspClient(self.url, self.transport, self.read_timeout)
                client.start()
                process = self._process = self._start_decoder()

                def feed(data):
                    process.stdin.write(data)

                gate = InBandParameterFilter(feed)
                stats = StreamStats(gate)
                LOG.info('RTP session %d playing (%s); waiting for an in-band SPS.', generation, self.transport)
                last_packet = next_keepalive = next_stats = time.monotonic()
                gate_started = None
                previous = (0, 0, 0, self._frame_id)
                while not self._stop.is_set():
                    now = time.monotonic()
                    if now >= next_keepalive:
                        client.keepalive()
                        next_keepalive = now + 20.
                    if self.stats_interval and now >= next_stats:
                        if now > last_packet - 1 and stats.packets:
                            current = (stats.packets, stats.lost, stats.pictures, self._frame_id)
                            LOG.info('RTP stats/%gs session %d: packets %d, lost %d, pictures %d, '
                                     'decoded frames %d; session totals: decode errors %d, framing resyncs %d.',
                                     self.stats_interval, generation, current[0] - previous[0],
                                     current[1] - previous[1], current[2] - previous[2],
                                     current[3] - previous[3], self._decode_errors, client.resyncs)
                            previous = current
                        next_stats = now + self.stats_interval
                    for packet in client.packets():
                        if packet is not None:
                            stats.add(packet, now)
                            last_packet = now
                    if time.monotonic() - last_packet > self.read_timeout:
                        raise TimeoutError(f'no RTP packets for {self.read_timeout:.0f}s')
                    with self._lock:
                        received_at = self._received_at
                    if gate.started and gate_started is None:
                        gate_started = now
                        LOG.info('RTP session %d: in-band SPS received after %d skipped NAL units; decoding.',
                                 generation, gate.skipped)
                    if gate_started is not None and now - (received_at or gate_started) > self.read_timeout:
                        raise TimeoutError(f'no decoded frames for {self.read_timeout:.0f}s')
            except (OSError, RuntimeError, ConnectionError, ValueError) as exc:
                if not self._stop.is_set():
                    LOG.warning('RTP session %d interrupted: %s; reconnecting.', generation, exc)
            finally:
                if client is not None:
                    client.close()
                self._stop_decoder(process)
            self._stop.wait(self.reconnect_delay)

    def read(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return frame, self._frame_id, self._received_at, self._generation

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.read_timeout + 5.)


def box_iou(a, b):
    left = np.maximum(a[:2], b[:2])
    right = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(right - left, 0)))
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0)))
    return intersection / max(area_a + area_b - intersection, 1e-9)


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    confidence: float
    class_id: int
    last_seen: float
    trail: deque = field(default_factory=deque)


class IoUTracker:
    """Class-aware greedy IoU association; no appearance model or motion prediction.

    IDs may change after occlusion or rapid camera movement. Only detections
    observed in the current processed frame are returned for drawing.
    """

    def __init__(self, iou_threshold=.3, max_age=1., trail_length=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.trail_length = trail_length
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()  # Keep IDs increasing across reconnections.

    def update(self, detections, now):
        self.tracks = {key: track for key, track in self.tracks.items()
                       if now - track.last_seen <= self.max_age}
        candidates = []
        for key, track in self.tracks.items():
            for index, det in enumerate(detections):
                if track.class_id == int(det[5]):
                    overlap = box_iou(track.box, det[:4])
                    if overlap >= self.iou_threshold:
                        candidates.append((overlap, key, index))
        matched_tracks, matched_detections, assignments = set(), set(), {}
        for _, key, index in sorted(candidates, reverse=True):
            if key not in matched_tracks and index not in matched_detections:
                assignments[index] = key
                matched_tracks.add(key)
                matched_detections.add(index)
        visible = []
        for index, det in enumerate(detections):
            key = assignments.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = Track(key, det[:4].copy(), float(det[4]), int(det[5]), now,
                                         deque(maxlen=self.trail_length))
            track = self.tracks[key]
            track.box = det[:4].copy()
            track.confidence = float(det[4])
            track.last_seen = now
            track.trail.append(tuple(((det[:2] + det[2:4]) / 2).astype(int)))
            visible.append(track)
        return visible


def draw_tracks(frame, tracks, names):
    for track in tracks:
        color = tuple(int(value) for value in np.random.default_rng(track.track_id).integers(80, 256, 3))
        x1, y1, x2, y2 = track.box.astype(int)
        name = names.get(track.class_id, str(track.class_id)) if isinstance(names, dict) else names[track.class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {track.track_id} {name} {track.confidence:.2f}',
                    (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
        if len(track.trail) > 1:
            cv2.polylines(frame, [np.asarray(track.trail, dtype=np.int32)], False, color, 2)


def waiting_screen(seconds=0.):
    """Display connection status instead of leaving an old frame on screen."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image, 'Waiting for Mambo video / reconnecting', (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f'No new frame for {seconds:.1f}s - q / Esc to quit', (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (180, 180, 180), 1, cv2.LINE_AA)
    return image


def run(opt):
    detector = None
    scale_boxes = None
    if not opt.preview_only:
        # Preview works with just OpenCV/numpy, without loading PyTorch or YOLO.
        from bebop_camera_inference import FFCAYoloDetector
        from utils.general import scale_boxes
        detector = FFCAYoloDetector(opt.weights, imgsz=opt.imgsz, conf_thres=opt.conf_thres,
                                    iou_thres=opt.iou_thres, device=opt.device, half=opt.half,
                                    data=opt.data, classes=opt.classes, max_det=opt.max_det)
        detector.load_model()

    if opt.backend == 'vlc':
        receiver = VlcVideoReceiver(opt.rtsp_url, opt.rtsp_transport, opt.vlc_width,
                                    opt.vlc_height, opt.network_caching, opt.read_timeout,
                                    opt.reconnect_delay, opt.stats_interval, opt.vlc_drop_late_frames)
    elif opt.backend == 'rtp':
        receiver = RtpVideoReceiver(opt.rtsp_url, opt.rtsp_transport, opt.rtp_width, opt.rtp_height,
                                    opt.read_timeout, opt.reconnect_delay, opt.stats_interval)
    else:
        receiver = MamboVideoReceiver(opt.rtsp_url, opt.rtsp_transport, opt.open_timeout,
                                      opt.read_timeout, opt.reconnect_delay)
    tracker = IoUTracker(opt.track_iou, opt.track_max_age, opt.trail_length)
    writer = None
    writer_shape = None
    processed_frames = 0
    try:
        receiver.start()
        started = last_new_frame = time.monotonic()
        last_id, generation = -1, -1
        fps, previous_frame_time = 0., None
        last_waiting_display = 0.
        stalled = False
        if not opt.no_view:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW, waiting_screen())
        while True:
            now = time.monotonic()
            if opt.duration and now - started >= opt.duration:
                LOG.info('Stopping: configured duration reached.')
                break
            # Pump UI even while frames are missing so q/Esc remain responsive.
            if not opt.no_view:
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    LOG.info('Stopping: q or Esc pressed.')
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    LOG.info('Stopping: preview window closed.')
                    break
            if receiver.error:
                raise RuntimeError(receiver.error)
            frame, frame_id, received_at, current_generation = receiver.read()
            if received_at is not None and received_at > last_new_frame:
                last_new_frame = received_at  # stream progress, possibly a static scene without new pixels
            if opt.frame_timeout > 0 and now - last_new_frame > opt.frame_timeout:
                raise RuntimeError('No new Mambo frames within the frame timeout. Check FPV camera/Wi-Fi, '
                                   'close other video clients, or try the other --rtsp-transport.')
            if frame is None or frame_id == last_id or now - last_new_frame > opt.read_timeout:
                if not stalled and now - last_new_frame > opt.read_timeout:
                    stalled = True
                    LOG.warning('No new frame for %.1fs; showing waiting screen (window stays open).',
                                now - last_new_frame)
                if not opt.no_view and now - last_new_frame > opt.read_timeout and now - last_waiting_display >= .2:
                    cv2.imshow(WINDOW, waiting_screen(now - last_new_frame))
                    last_waiting_display = now
                time.sleep(.005)
                continue
            if stalled:
                stalled = False
                LOG.info('Video resumed.')
            last_id = frame_id
            processed_frames += 1
            if processed_frames == 1:
                LOG.info('First decoded frame received: %dx%d.', frame.shape[1], frame.shape[0])
            if generation != current_generation:
                tracker.reset()
                generation = current_generation
                previous_frame_time, fps = None, 0.
                LOG.info('Receiving frames from stream session %d.', generation)
            if previous_frame_time is not None:
                instantaneous_fps = 1. / max(now - previous_frame_time, 1e-6)
                fps = instantaneous_fps if not fps else .9 * fps + .1 * instantaneous_fps
            previous_frame_time = now
            latency = 0.
            if detector is not None:
                det, input_shape, latency, _ = detector.infer(frame)
                # infer() runs under torch.inference_mode(); scale_boxes() edits in place,
                # which PyTorch forbids on inference tensors, so work on a normal copy.
                det = det.clone()
                if len(det):
                    det[:, :4] = scale_boxes(input_shape, det[:, :4], frame.shape).round()
                tracks = tracker.update(det.detach().cpu().numpy(), received_at)
                draw_tracks(frame, tracks, detector.names)
            status = f'{"PREVIEW" if detector is None else "TRACKING"}  FPS: {fps:.1f}  Inference: {latency:.1f} ms'
            cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            if opt.save_video:
                h, w = frame.shape[:2]
                if writer is None:
                    Path(opt.output).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*'mp4v'), opt.output_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot open output video: {opt.output}')
                    writer_shape = (h, w)
                if (h, w) != writer_shape:
                    raise RuntimeError('Stream resolution changed while recording; restart recording.')
                writer.write(frame)
            if not opt.no_view:
                cv2.imshow(WINDOW, frame)
    except KeyboardInterrupt:
        LOG.info('Stopped by user.')
    finally:
        receiver.stop()
        LOG.info('Processed %d decoded frames.', processed_frames)
        if writer is not None:
            writer.release()
        if not opt.no_view:
            cv2.destroyAllWindows()


def parse_opt(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preview-only', action='store_true', help='show video without loading YOLO')
    parser.add_argument('--backend', choices=('opencv', 'vlc', 'rtp'), default='rtp',
                        help='rtp ignores the Mambo SDP SPS/PPS, which do not match its stream (recommended)')
    parser.add_argument('--rtp-width', type=int, default=640, help='RTP backend output width')
    parser.add_argument('--rtp-height', type=int, default=360, help='RTP backend output height')
    parser.add_argument('--vlc-width', type=int, default=0, help='VLC output width; 0 keeps decoded size')
    parser.add_argument('--vlc-height', type=int, default=0, help='VLC output height; 0 keeps decoded size')
    parser.add_argument('--network-caching', type=int, default=300, help='VLC buffer in milliseconds')
    parser.add_argument('--vlc-drop-late-frames', action='store_true',
                        help="restore VLC's default of dropping frames it considers late")
    parser.add_argument('--stats-interval', type=float, default=5.,
                        help='log VLC receive/decode/display counters every N seconds; 0 disables')
    parser.add_argument('--rtsp-url', default=DEFAULT_URL)
    parser.add_argument('--rtsp-transport', choices=('udp', 'tcp'), default='udp')
    parser.add_argument('--open-timeout', type=float, default=5.)
    parser.add_argument('--read-timeout', type=float, default=3.)
    parser.add_argument('--reconnect-delay', type=float, default=1.)
    parser.add_argument('--frame-timeout', type=float, default=0.,
                        help='0 keeps waiting/reconnecting; positive value exits after this many seconds without a NEW frame')
    parser.add_argument('--weights', nargs='+', help='required for tracking; local trained model path(s)')
    parser.add_argument('--data', default=str(Path(__file__).parent / 'data/coco128.yaml'))
    parser.add_argument('--device', default='', help='auto, CUDA index 0, or cpu')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=.25)
    parser.add_argument('--iou-thres', type=float, default=.45)
    parser.add_argument('--classes', nargs='+', type=int, help='track only these model class IDs')
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--track-iou', type=float, default=.3)
    parser.add_argument('--track-max-age', type=float, default=1., help='retain missing IDs for this many seconds')
    parser.add_argument('--trail-length', type=int, default=30)
    parser.add_argument('--no-view', action='store_true', help='disable preview window')
    parser.add_argument('--duration', type=float, default=0., help='seconds after receiver start; 0 means unlimited')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--output', default='outputs/mambo_tracking.mp4')
    parser.add_argument('--output-fps', type=float, default=30., help='fixed saved-video playback FPS')
    opt = parser.parse_args(argv)
    if not opt.preview_only and not opt.weights:
        parser.error('--weights is required for tracking, or use --preview-only')
    if not opt.preview_only:
        for weight in opt.weights:
            if not Path(weight).is_file():
                parser.error(f'weight file not found: {weight}')
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    if len(opt.imgsz) != 2 or any(size <= 0 for size in opt.imgsz):
        parser.error('--imgsz requires one or two positive integers')
    for name in ('open_timeout', 'read_timeout', 'reconnect_delay', 'track_max_age', 'output_fps'):
        if not math.isfinite(getattr(opt, name)) or getattr(opt, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and positive')
    if not math.isfinite(opt.frame_timeout) or opt.frame_timeout < 0:
        parser.error('--frame-timeout must be finite and nonnegative')
    for name in ('conf_thres', 'iou_thres', 'track_iou'):
        if not 0 < getattr(opt, name) <= 1:
            parser.error(f'--{name.replace("_", "-")} must be in (0, 1]')
    if not math.isfinite(opt.duration) or opt.duration < 0 or opt.trail_length < 1 or opt.max_det < 1:
        parser.error('duration must be finite and nonnegative; trail-length and max-det must be positive')
    if opt.vlc_width < 0 or opt.vlc_height < 0 or (opt.vlc_width == 0) != (opt.vlc_height == 0):
        parser.error('--vlc-width and --vlc-height must both be 0 (decoded size) or both positive')
    if opt.rtp_width < 2 or opt.rtp_height < 2:
        parser.error('--rtp-width and --rtp-height must be at least 2')
    if opt.network_caching < 0:
        parser.error('--network-caching must be nonnegative')
    if not math.isfinite(opt.stats_interval) or opt.stats_interval < 0:
        parser.error('--stats-interval must be finite and nonnegative')
    url = urlsplit(opt.rtsp_url)
    if url.scheme.lower() != 'rtsp' or not url.hostname:
        parser.error('--rtsp-url must be an RTSP URL with a host')
    return opt


if __name__ == '__main__':
    # A native crash (e.g. inside libVLC/OpenCV) then prints a traceback instead of exiting silently.
    faulthandler.enable()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    try:
        run(parse_opt())
    except (RuntimeError, ImportError, cv2.error) as exc:
        LOG.error('%s', exc)
        raise SystemExit(1)
