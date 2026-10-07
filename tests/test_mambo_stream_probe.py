"""RTP loss accounting and H.264 depacketization checks; no drone required."""

import io
import struct
import unittest

import mambo_stream_probe as probe


def rtp(seq, timestamp, payload, marker=0):
    return struct.pack('!BBHII', 0x80, (marker << 7) | 96, seq & 0xFFFF, timestamp, 1) + payload


def fu_a(nal, size=4):
    head, body = nal[0], nal[1:]
    chunks = [body[i:i + size] for i in range(0, len(body), size)]
    return [bytes([(head & 0xE0) | 28, (0x80 if i == 0 else 0) | (0x40 if i == len(chunks) - 1 else 0)
                   | (head & 0x1F)]) + chunk for i, chunk in enumerate(chunks)]


class ProbeTests(unittest.TestCase):
    def test_fu_a_reassembles_to_annex_b(self):
        dump = io.BytesIO()
        stats = probe.StreamStats(dump)
        idr = bytes([0x65]) + bytes(range(1, 11))
        for seq, payload in enumerate(fu_a(idr)):
            stats.add(rtp(seq, 0, payload, marker=payload[1] & 0x40 != 0), seq * .01)
        self.assertEqual(dump.getvalue(), probe.START_CODE + idr)
        self.assertEqual((stats.frames, stats.lost, stats.nal_types[5], stats.fu_incomplete), (1, 0, 1, 0))

    def test_sequence_gap_counts_loss_and_drops_incomplete_nal(self):
        dump = io.BytesIO()
        stats = probe.StreamStats(dump)
        packets = fu_a(bytes([0x41]) + bytes(12))
        for seq, payload in enumerate(packets):
            if seq != 1:  # lost middle fragment
                stats.add(rtp(seq, 0, payload, marker=seq == len(packets) - 1), 0.)
        self.assertEqual((stats.lost, stats.loss_events, stats.damaged_frames), (1, 1, 1))
        self.assertGreaterEqual(stats.fu_incomplete, 1)
        self.assertEqual(dump.getvalue(), b'')

    def test_sequence_wraparound_is_not_loss(self):
        stats = probe.StreamStats()
        stats.add(rtp(0xFFFF, 0, b'\x41a', marker=1), 0.)
        stats.add(rtp(0, 3000, b'\x41b', marker=1), .03)
        self.assertEqual((stats.lost, stats.reordered, stats.frames), (0, 0, 2))

    def test_arrival_stall_is_recorded(self):
        stats = probe.StreamStats()
        stats.add(rtp(0, 0, b'\x41a', marker=1), 10.)
        stats.add(rtp(1, 3000, b'\x41b', marker=1), 12.5)
        self.assertEqual(stats.stalls, [(0., 2.5)])

    def test_non_rtp_packet_is_ignored(self):
        self.assertIsNone(probe.parse_rtp(b'\x00' * 12))


if __name__ == '__main__':
    unittest.main()
