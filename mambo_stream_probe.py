"""Measure the Mambo RTSP/RTP H.264 stream layer by layer, without decoding it live.

Run it alone (close other Mambo video clients first) so results are not mixed:
    python mambo_stream_probe.py --seconds 60 --transport tcp

It reports, separately:
1. RTP packets actually received and sequence-number gaps (packets lost before the
   PC received them; over TCP these were dropped on the drone side, not on the link)
2. arrival stalls (no packets for a while) and whether the server ended the session
3. H.264 structure: frames, IDR interval, incomplete FU-A fragments, SPS/PPS
4. optionally, an offline FFmpeg decode of the captured bitstream (--dump)
This program only receives video; it sends no flight commands.
"""

import argparse
import base64
from collections import Counter
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import time
from urllib.parse import urljoin, urlsplit

DEFAULT_URL = 'rtsp://192.168.99.1/media/stream2'
START_CODE = b'\x00\x00\x00\x01'
NAL_NAMES = {1: 'non-IDR', 5: 'IDR', 6: 'SEI', 7: 'SPS', 8: 'PPS', 9: 'AUD'}


def parse_rtp(packet):
    """Return (payload_type, seq, timestamp, marker, payload) or None for a non-RTP packet."""
    if len(packet) < 12 or packet[0] >> 6 != 2:
        return None
    csrc, extension, padding = packet[0] & 0x0F, packet[0] & 0x10, packet[0] & 0x20
    marker, payload_type = packet[1] >> 7, packet[1] & 0x7F
    seq, timestamp = struct.unpack('!HI', packet[2:8])
    offset = 12 + 4 * csrc
    if extension:
        if len(packet) < offset + 4:
            return None
        offset += 4 + 4 * struct.unpack('!H', packet[offset + 2:offset + 4])[0]
    end = len(packet) - (packet[-1] if padding else 0)
    if offset > end:
        return None
    return payload_type, seq, timestamp, marker, packet[offset:end]


class StreamStats:
    """RTP sequence/arrival statistics plus RFC 6184 H.264 depacketization."""

    def __init__(self, dump=None):
        self.dump = dump
        self.packets = self.payload_bytes = self.lost = self.loss_events = 0
        self.duplicates = self.reordered = self.max_gap = 0
        self.max_arrival_gap = 0.
        self.stalls = []  # (seconds into capture, stall length)
        self.nal_types = Counter()
        self.fu_incomplete = self.frames = self.damaged_frames = 0
        self.idr_times = []
        # Pictures counted from slice headers (first_mb_in_slice == 0), independent of RTP timing.
        self.pictures = self.markers = self.timestamp_changes = 0
        self.timestamp_deltas = Counter()  # 90 kHz ticks between consecutive RTP timestamps
        self._seq = self._timestamp = self._last_arrival = self._first_arrival = None
        self._fu = None
        self._last_timestamp = None
        self._frame_damaged = False
        self._stall_threshold = 1.

    def add(self, packet, now):
        rtp = parse_rtp(packet)
        if rtp is None:
            return
        _, seq, timestamp, marker, payload = rtp
        if self._first_arrival is None:
            self._first_arrival = now
        if self._last_arrival is not None:
            gap = now - self._last_arrival
            self.max_arrival_gap = max(self.max_arrival_gap, gap)
            if gap >= self._stall_threshold:
                self.stalls.append((self._last_arrival - self._first_arrival, gap))
        self._last_arrival = now
        self.packets += 1
        self.payload_bytes += len(payload)
        if self._seq is not None:
            delta = (seq - self._seq) & 0xFFFF
            if delta == 0:
                self.duplicates += 1
                return
            if delta >= 0x8000:
                self.reordered += 1
                return
            if delta > 1:
                self.lost += delta - 1
                self.loss_events += 1
                self.max_gap = max(self.max_gap, delta - 1)
                self._frame_damaged = True
                if self._fu is not None:
                    self._fu = None
                    self.fu_incomplete += 1
        self._seq = seq
        if self._last_timestamp is not None and timestamp != self._last_timestamp:
            self.timestamp_changes += 1
            self.timestamp_deltas[(timestamp - self._last_timestamp) & 0xFFFFFFFF] += 1
        self._last_timestamp = timestamp
        self.markers += marker
        if self._timestamp is not None and timestamp != self._timestamp:
            self._end_frame()
        self._timestamp = timestamp
        self._depacketize(payload, now)
        if marker:
            self._end_frame()
            self._timestamp = None

    def _end_frame(self):
        self.frames += 1
        self.damaged_frames += self._frame_damaged
        self._frame_damaged = False

    def _nal(self, nal, now):
        nal_type = nal[0] & 0x1F
        self.nal_types[nal_type] += 1
        if nal_type in (1, 5) and len(nal) > 1 and nal[1] & 0x80:  # ue(v) first_mb_in_slice == 0
            self.pictures += 1
            if nal_type == 5:
                self.idr_times.append(now)
        if self.dump:
            self.dump.write(START_CODE + nal)

    def _depacketize(self, payload, now):
        if not payload:
            return
        nal_type = payload[0] & 0x1F
        if 1 <= nal_type <= 23:
            self._nal(payload, now)
        elif nal_type == 24:  # STAP-A
            offset = 1
            while offset + 2 <= len(payload):
                size = struct.unpack('!H', payload[offset:offset + 2])[0]
                self._nal(payload[offset + 2:offset + 2 + size], now)
                offset += 2 + size
        elif nal_type == 28 and len(payload) >= 2:  # FU-A
            start, end = payload[1] & 0x80, payload[1] & 0x40
            if start:
                if self._fu is not None:
                    self.fu_incomplete += 1
                self._fu = bytearray([(payload[0] & 0xE0) | (payload[1] & 0x1F)])
            elif self._fu is None:
                self.fu_incomplete += 1  # middle/end fragment without its start
                self._frame_damaged = True
                return
            self._fu += payload[2:]
            if end:
                self._nal(bytes(self._fu), now)
                self._fu = None
        else:
            self.nal_types[f'unsupported {nal_type}'] += 1

    def summary(self, elapsed):
        expected = self.packets + self.lost
        idr = [b - a for a, b in zip(self.idr_times, self.idr_times[1:])]
        lines = [
            f'Capture: {elapsed:.1f}s, RTP packets {self.packets}, payload {self.payload_bytes / 1024:.0f} kB '
            f'({self.payload_bytes * 8 / 1000 / max(elapsed, 1e-9):.0f} kbit/s)',
            f'RTP sequence: lost {self.lost} of {expected} ({100 * self.lost / max(expected, 1):.2f}%) in '
            f'{self.loss_events} gaps (largest {self.max_gap}); duplicates {self.duplicates}, out-of-order {self.reordered}',
            f'Arrival: longest gap between packets {self.max_arrival_gap:.2f}s; '
            f'stalls >= {self._stall_threshold:.0f}s: {len(self.stalls)}'
            + ''.join(f'\n  at {at:.1f}s for {length:.1f}s' for at, length in self.stalls[:10]),
            f'H.264 pictures (slice headers): {self.pictures} ({self.pictures / max(elapsed, 1e-9):.1f}/s); '
            f'RTP frames by timestamp/marker {self.frames}, with missing packets {self.damaged_frames}; '
            f'incomplete FU-A {self.fu_incomplete}',
            f'RTP timing: marker bits {self.markers}, timestamp changes {self.timestamp_changes}; most common '
            f'deltas (90 kHz ticks: count) ' + ', '.join(f'{d}: {c}' for d, c in self.timestamp_deltas.most_common(5)),
            'NAL units: ' + ', '.join(f'{NAL_NAMES.get(k, k)}={v}' for k, v in sorted(self.nal_types.items(), key=str)),
            f'IDR pictures {len(self.idr_times)}' + (f', interval {min(idr):.2f}-{max(idr):.2f}s (mean '
                                                   f'{sum(idr) / len(idr):.2f}s)' if idr else ''),
        ]
        return '\n'.join(lines)


class RtspClient:
    def __init__(self, url, transport, timeout):
        self.url, self.transport, self.timeout = url, transport, timeout
        parts = urlsplit(url)
        self.sock = socket.create_connection((parts.hostname, parts.port or 554), timeout=timeout)
        self.cseq = 0
        self.session = None
        self.buffer = b''
        self.rtp_socket = None
        self.bytes_received = 0
        self.channel_bytes = Counter()
        self.setup_transport = ''
        self.resyncs = self.skipped_bytes = 0
        self.desync_samples = []

    def request(self, method, url, headers=None):
        self.cseq += 1
        lines = [f'{method} {url} RTSP/1.0', f'CSeq: {self.cseq}', 'User-Agent: mambo-stream-probe']
        if self.session:
            lines.append(f'Session: {self.session}')
        lines += [f'{k}: {v}' for k, v in (headers or {}).items()]
        self.sock.sendall(('\r\n'.join(lines) + '\r\n\r\n').encode())
        return self._response()

    def _read_more(self):
        data = self.sock.recv(65536)
        if not data:
            raise ConnectionError('RTSP server closed the connection')
        self.bytes_received += len(data)
        self.buffer += data

    def _response(self):
        while True:
            # Skip interleaved RTP that may arrive before the response.
            while self.buffer.startswith(b'$'):
                if len(self.buffer) < 4:
                    self._read_more()
                    continue
                size = struct.unpack('!H', self.buffer[2:4])[0]
                if len(self.buffer) < 4 + size:
                    self._read_more()
                    continue
                self.buffer = self.buffer[4 + size:]
            end = self.buffer.find(b'\r\n\r\n')
            if self.buffer.startswith(b'RTSP/') and end >= 0:
                head = self.buffer[:end].decode(errors='replace')
                headers = {k.strip().lower(): v.strip() for k, _, v in
                           (line.partition(':') for line in head.split('\r\n')[1:])}
                length = int(headers.get('content-length', 0))
                if len(self.buffer) < end + 4 + length:
                    self._read_more()
                    continue
                body = self.buffer[end + 4:end + 4 + length].decode(errors='replace')
                self.buffer = self.buffer[end + 4 + length:]
                status = int(head.split()[1])
                return status, headers, body
            self._read_more()

    def start(self):
        status, headers, sdp = self.request('DESCRIBE', self.url, {'Accept': 'application/sdp'})
        if status != 200:
            raise RuntimeError(f'DESCRIBE returned {status}')
        base = headers.get('content-base', self.url)
        if not base.endswith('/'):
            base += '/'
        video = sdp.split('m=video', 1)[1].split('\nm=', 1)[0] if 'm=video' in sdp else ''
        control = re.search(r'a=control:(\S+)', video)
        control = control.group(1) if control else ''
        track = control if control.startswith('rtsp://') else urljoin(base, control) if control else self.url
        sprop = re.search(r'sprop-parameter-sets=([^;\s]+)', video)
        parameter_sets = [base64.b64decode(item + '=' * (-len(item) % 4))
                          for item in sprop.group(1).split(',') if item] if sprop else []
        if self.transport == 'tcp':
            transport = 'RTP/AVP/TCP;unicast;interleaved=0-1'
        else:
            self.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rtp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
            for port in range(50000, 50100, 2):
                try:
                    self.rtp_socket.bind(('', port))
                    break
                except OSError:
                    continue
            self.rtp_socket.settimeout(.5)
            transport = f'RTP/AVP;unicast;client_port={port}-{port + 1}'
        status, headers, _ = self.request('SETUP', track, {'Transport': transport})
        if status != 200:
            raise RuntimeError(f'SETUP returned {status} for {transport}')
        self.session = headers.get('session', '').split(';')[0]
        self.setup_transport = headers.get('transport', '(none)')
        # Like VLC/FFmpeg, PLAY the aggregate (Content-Base) URL.
        status, _, _ = self.request('PLAY', base, {'Range': 'npt=0.000-'})
        if status != 200:
            raise RuntimeError(f'PLAY returned {status}')
        return sdp, parameter_sets

    def packets(self):
        """Yield RTP packets until the deadline managed by the caller."""
        if self.rtp_socket is not None:
            try:
                yield self.rtp_socket.recv(65536)
            except socket.timeout:
                yield None
            return
        while True:
            if self.buffer.startswith(b'$') and len(self.buffer) >= 4:
                channel, size = self.buffer[1], struct.unpack('!H', self.buffer[2:4])[0]
                if len(self.buffer) >= 4 + size:
                    packet, self.buffer = self.buffer[4:4 + size], self.buffer[4 + size:]
                    self.channel_bytes[channel] += size
                    if channel == 0:
                        yield packet
                    continue
            elif self.buffer.startswith(b'RTSP/') and b'\r\n\r\n' in self.buffer:
                self._response()  # keep-alive reply
                continue
            elif self.buffer and not self.buffer.startswith(b'$') and not (
                    self.buffer.startswith(b'RTSP/') or b'RTSP/'.startswith(self.buffer[:5])):
                self._resync()
                continue
            try:
                self._read_more()
            except socket.timeout:
                yield None
            return

    def _resync(self):
        """Skip bytes that are not '$'-framed; keep evidence instead of hiding it."""
        if len(self.desync_samples) < 3:
            self.desync_samples.append(self.buffer[:48].hex(' '))
        index = 1
        while True:
            index = self.buffer.find(b'$', index)
            if index < 0 or (index + 1 < len(self.buffer) and self.buffer[index + 1] <= 3):
                break
            index += 1
        skip = len(self.buffer) if index < 0 else index
        self.resyncs += 1
        self.skipped_bytes += skip
        self.buffer = self.buffer[skip:]

    def keepalive(self):
        # Reply is consumed by packets()/_response; do not wait for it here.
        self.cseq += 1
        self.sock.sendall(f'OPTIONS {self.url} RTSP/1.0\r\nCSeq: {self.cseq}\r\nSession: {self.session}\r\n\r\n'.encode())

    def close(self):
        try:
            self.cseq += 1
            self.sock.sendall(f'TEARDOWN {self.url} RTSP/1.0\r\nCSeq: {self.cseq}\r\n'
                              f'Session: {self.session}\r\n\r\n'.encode())
        except OSError:
            pass
        self.sock.close()
        if self.rtp_socket is not None:
            self.rtp_socket.close()


def find_ffmpeg():
    found = shutil.which('ffmpeg')
    if found:
        return found
    root = Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft' / 'WinGet' / 'Packages'
    return next((str(p) for p in root.glob('Gyan.FFmpeg*/*/bin/ffmpeg.exe')), None) if root.is_dir() else None


def decode_check(path):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return 'Offline decode: skipped (ffmpeg not found).'
    result = subprocess.run([ffmpeg, '-hide_banner', '-v', 'error', '-f', 'h264', '-i', str(path),
                             '-f', 'null', '-'], capture_output=True, text=True, errors='replace')
    errors = [line for line in result.stderr.splitlines() if line.strip()]
    return (f'Offline decode of {path}: {len(errors)} FFmpeg error lines'
            + ''.join(f'\n  {line}' for line in errors[:8]))


def run(opt):
    dump = open(opt.dump, 'wb') if opt.dump else None
    stats = StreamStats(dump)
    client = RtspClient(opt.rtsp_url, opt.transport, opt.timeout)
    ended = None
    try:
        sdp, parameter_sets = client.start()
        print('SDP video section:\n  ' + '\n  '.join(line for line in sdp.splitlines()
                                                     if line.startswith(('m=video', 'a=rtpmap', 'a=fmtp'))))
        if dump:
            for item in parameter_sets:
                dump.write(START_CODE + item)
        print(f'SETUP reply Transport: {client.setup_transport}')
        print(f'PLAY accepted ({opt.transport}); capturing {opt.seconds:.0f}s. Ctrl+C stops early.')
        started = next_report = time.monotonic()
        next_keepalive = started + opt.keepalive
        while (now := time.monotonic()) - started < opt.seconds:
            if now >= next_keepalive:
                client.keepalive()
                next_keepalive = now + opt.keepalive
            if now >= next_report:
                print(f'  {now - started:5.1f}s: packets {stats.packets}, lost {stats.lost}, frames {stats.frames}, '
                      f'socket bytes {client.bytes_received}, interleaved by channel {dict(client.channel_bytes)}')
                next_report = now + 5.
            for packet in client.packets():
                if packet is not None:
                    stats.add(packet, time.monotonic())
                elif time.monotonic() - (stats._last_arrival or started) > opt.timeout:
                    raise TimeoutError(f'no RTP packets for {opt.timeout:.0f}s')
    except KeyboardInterrupt:
        ended = 'stopped by user'
    except (OSError, RuntimeError, ConnectionError) as exc:
        ended = f'{type(exc).__name__}: {exc}'
    finally:
        client.close()
        if dump:
            dump.close()
    elapsed = (stats._last_arrival or 0) - (stats._first_arrival or 0)
    print('\n' + stats.summary(elapsed))
    print(f'RTSP socket bytes: {client.bytes_received}; interleaved bytes by channel: {dict(client.channel_bytes)}')
    print(f'TCP interleaved framing: {client.resyncs} resyncs, {client.skipped_bytes} bytes outside $-frames'
          + ''.join(f'\n  sample: {sample}' for sample in client.desync_samples))
    print('Session end: ' + (ended or 'capture duration reached'))
    if opt.dump and stats.pictures:
        print(decode_check(opt.dump))
    elif opt.dump:
        print('Offline decode: skipped (no pictures captured; the dump holds only SDP parameter sets).')
    print('\nHow to read this: lost RTP packets over TCP were dropped before leaving the drone; the dump '
          'keeps only fully received NAL units, so offline decode errors there mean the bitstream itself is '
          'damaged; both clean while the live player shows errors points at the PC playback pipeline.')


def parse_opt(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rtsp-url', default=DEFAULT_URL)
    parser.add_argument('--transport', choices=('tcp', 'udp'), default='tcp',
                        help='udp may need a Windows Firewall allowance for Python')
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--timeout', type=float, default=5., help='connect / no-packet timeout')
    parser.add_argument('--keepalive', type=float, default=20., help='RTSP OPTIONS interval in seconds')
    parser.add_argument('--dump', help='write received H.264 (Annex B) here and decode it offline with FFmpeg')
    opt = parser.parse_args(argv)
    if opt.seconds <= 0 or opt.timeout <= 0 or opt.keepalive <= 0:
        parser.error('seconds, timeout and keepalive must be positive')
    if opt.dump:
        Path(opt.dump).parent.mkdir(parents=True, exist_ok=True)
    return opt


if __name__ == '__main__':
    run(parse_opt())
