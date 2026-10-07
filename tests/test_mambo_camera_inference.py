"""Tracking identity and stream-stall regression checks; no drone required."""

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

import mambo_camera_inference as mambo


def detection(x=0, cls=0):
    return np.array([[x, 0, x + 100, 100, .9, cls]], dtype=float)


class TrackerTests(unittest.TestCase):
    def test_motion_and_short_occlusion_keep_identity(self):
        tracker = mambo.IoUTracker(max_age=1., trail_length=2)
        first = tracker.update(detection(), 0.)[0].track_id
        self.assertEqual(tracker.update(detection(5), .1)[0].track_id, first)
        self.assertEqual(tracker.update(np.empty((0, 6)), .3), [])
        track = tracker.update(detection(10), .5)[0]
        self.assertEqual(track.track_id, first)
        self.assertEqual(len(track.trail), 2)

    def test_expiry_and_reconnection_allocate_new_ids(self):
        tracker = mambo.IoUTracker(max_age=1.)
        first = tracker.update(detection(), 0.)[0].track_id
        second = tracker.update(detection(), 1.1)[0].track_id
        self.assertNotEqual(first, second)
        tracker.reset()
        self.assertGreater(tracker.update(detection(), 1.2)[0].track_id, second)

    def test_class_change_does_not_reuse_id(self):
        tracker = mambo.IoUTracker()
        first = tracker.update(detection(cls=0), 0.)[0].track_id
        self.assertNotEqual(tracker.update(detection(cls=1), .1)[0].track_id, first)

    def test_assignment_is_one_to_one(self):
        tracker = mambo.IoUTracker()
        first = tracker.update(detection(), 0.)[0].track_id
        result = tracker.update(np.concatenate([detection(1), detection(2)]), .1)
        self.assertEqual(len({t.track_id for t in result}), 2)
        self.assertIn(first, [t.track_id for t in result])


class StreamTests(unittest.TestCase):
    def test_default_waiting_does_not_exit_without_frames(self):
        opt = mambo.parse_opt(['--preview-only', '--no-view', '--duration', '0.03'])
        self.assertEqual(opt.frame_timeout, 0.)
        with patch.object(mambo.MamboVideoReceiver, 'start'), patch.object(mambo.MamboVideoReceiver, 'stop'):
            mambo.run(opt)

    def test_preview_does_not_import_detector(self):
        opt = mambo.parse_opt(['--preview-only', '--no-view', '--duration', '0.01'])
        with patch.object(mambo.MamboVideoReceiver, 'start'), patch.object(mambo.MamboVideoReceiver, 'stop'):
            mambo.run(opt)

    def test_cached_frame_cannot_hide_stream_stall(self):
        opt = mambo.parse_opt(['--preview-only', '--no-view', '--frame-timeout', '0.02'])
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch.object(mambo.MamboVideoReceiver, 'start'), patch.object(mambo.MamboVideoReceiver, 'stop'), \
                patch.object(mambo.MamboVideoReceiver, 'read', return_value=(frame, 1, mambo.time.monotonic(), 1)):
            with self.assertRaisesRegex(RuntimeError, 'No new Mambo frames'):
                mambo.run(opt)


class InBandParameterFilterTests(unittest.TestCase):
    def test_nal_units_before_first_in_band_sps_are_dropped(self):
        written = []
        gate = mambo.InBandParameterFilter(written.append)
        nal = lambda *data: mambo.START_CODE + bytes(data)
        for data in (nal(0x68, 0xCE), nal(0x41, 0x80), nal(0x41, 0x00)):  # PPS + slices without an SPS
            gate.write(data)
        self.assertEqual((written, gate.skipped), ([], 3))
        for data in (nal(0x67, 0x64), nal(0x68, 0xCE), nal(0x65, 0x88)):  # in-band SPS/PPS + IDR
            gate.write(data)
        self.assertEqual(written, [nal(0x67, 0x64), nal(0x68, 0xCE), nal(0x65, 0x88)])


class VlcReconnectTests(unittest.TestCase):
    def test_pixel_buffer_and_rows_are_aligned_even_for_odd_width(self):
        receiver = mambo.VlcVideoReceiver('rtsp://test/stream', width=161, height=121)
        self.assertEqual(receiver._pixel_address % 32, 0)
        self.assertEqual(receiver._pitch % 32, 0)
        self.assertGreaterEqual(receiver._pitch, 161 * 4)

    def receiver(self, state='playing'):
        receiver = mambo.VlcVideoReceiver('rtsp://test/stream', read_timeout=3., reconnect_delay=1.)
        receiver._vlc = SimpleNamespace(State=SimpleNamespace(Error='error', Ended='ended', Stopped='stopped'))
        receiver._player = Mock()
        receiver._player.get_state.return_value = state
        receiver._player.video_get_size.return_value = (10, 10)
        receiver._media = Mock()
        receiver._attempt_started = 0.
        receiver._received_at = 1.
        receiver._frame = np.zeros((10, 10, 3), dtype=np.uint8)
        return receiver

    def test_stalled_playing_stream_clears_frame_and_waits_before_retry(self):
        receiver = self.receiver()
        player = receiver._player
        with patch.object(mambo.time, 'monotonic', return_value=5.):
            receiver._check_connection(5.)
        frame, *_ = receiver.read()
        self.assertIsNone(frame)
        player.stop.assert_called_once()
        player.release.assert_called_once()
        self.assertEqual(receiver._retry_at, 6.)
        self.assertIsNone(receiver.error)
        with patch.object(receiver, '_open_player') as reopen:
            receiver._check_connection(5.5)
            reopen.assert_not_called()
            receiver._check_connection(6.)
            reopen.assert_called_once()

    def test_read_never_stops_player_on_ui_thread(self):
        receiver = self.receiver()
        with patch.object(mambo.time, 'monotonic', return_value=100.):
            frame, *_ = receiver.read()
        self.assertIsNotNone(frame)
        receiver._player.stop.assert_not_called()

    def test_identical_rerender_is_not_a_new_frame(self):
        receiver = self.receiver()
        with patch.object(mambo.time, 'monotonic', return_value=2.):
            receiver._publish(np.ones((10, 10, 3), dtype=np.uint8))
        with patch.object(mambo.time, 'monotonic', return_value=4.5):
            receiver._publish(np.ones((10, 10, 3), dtype=np.uint8))
        _, frame_id, alive_at, _ = receiver.read()
        self.assertEqual((frame_id, alive_at, receiver._repeats), (1, 2., 1))
        receiver._check_connection(5.1)  # only re-renders for 3.1 s: treated as a stall
        self.assertEqual(receiver._reconnect_count, 1)

    def test_decoder_progress_keeps_static_scene_alive(self):
        receiver = self.receiver()
        receiver._progress_at = 4.5  # VLC displayed a new picture identical to the last one
        receiver._check_connection(5.)
        self.assertEqual(receiver._reconnect_count, 0)
        self.assertEqual(receiver.read()[2], 4.5)

    def test_failed_reconnect_is_retried_instead_of_exiting(self):
        receiver = self.receiver()
        receiver._player = receiver._media = None
        receiver._retry_at = 1.
        with patch.object(receiver, '_open_player', side_effect=RuntimeError('play failed')):
            receiver._check_connection(1.)
        self.assertIsNone(receiver.error)
        self.assertEqual(receiver._retry_at, 2.)

    def test_ended_stream_reconnects_even_with_recent_frame(self):
        receiver = self.receiver('ended')
        receiver._check_connection(1.1)
        self.assertEqual(receiver._reconnect_count, 1)
        self.assertIsNone(receiver.error)

    def test_healthy_stream_is_not_restarted(self):
        receiver = self.receiver()
        player = receiver._player
        receiver._check_connection(2.)
        frame, *_ = receiver.read()
        self.assertIsNotNone(frame)
        player.stop.assert_not_called()

    def test_reopened_session_increments_generation_and_clears_old_timestamp(self):
        receiver = self.receiver()
        receiver._instance = Mock()
        receiver._callbacks = (None, None, None)
        receiver._format_callbacks = (None, None)
        receiver._set_format_callbacks = Mock()
        receiver._instance.media_player_new.return_value.play.return_value = 0
        receiver._open_player()
        self.assertEqual(receiver._generation, 1)
        self.assertIsNone(receiver._received_at)
        self.assertIsNone(receiver._frame)
        receiver.stop()


if __name__ == '__main__':
    unittest.main()
