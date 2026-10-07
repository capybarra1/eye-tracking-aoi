import unittest
from scripts.supervised_aoi_server import step_preview


class FakeSession:
    def __init__(self, stop_at=99):
        self.running = True
        self.frames = []
        self.stop_at = stop_at

    def step(self, token):
        if token != 'current' or not self.running:
            raise ValueError('stale')
        self.frames.append(len(self.frames) + 1)
        if len(self.frames) == self.stop_at:
            self.running = False


class PreviewTests(unittest.TestCase):
    def test_batch_processes_every_frame(self):
        session = FakeSession()
        step_preview(session, 'current', 4)
        self.assertEqual(session.frames, [1, 2, 3, 4])

    def test_batch_stops_at_guard(self):
        session = FakeSession(stop_at=2)
        step_preview(session, 'current', 4)
        self.assertEqual(session.frames, [1, 2])

    def test_high_speed_processes_all_frames_and_stops_at_each_guard(self):
        for batch in (6, 10):
            session = FakeSession()
            step_preview(session, 'current', batch)
            self.assertEqual(session.frames, list(range(1, batch + 1)))
            for stop in range(1, batch + 1):
                session = FakeSession(stop_at=stop)
                step_preview(session, 'current', batch)
                self.assertEqual(session.frames, list(range(1, stop + 1)))

    def test_invalid_and_stale_requests_do_not_advance(self):
        for frames in (0, 11, True, 1.5):
            session = FakeSession()
            with self.assertRaises(ValueError):
                step_preview(session, 'current', frames)
            self.assertEqual(session.frames, [])
        session = FakeSession()
        with self.assertRaises(ValueError):
            step_preview(session, 'old', 4)
        self.assertEqual(session.frames, [])
