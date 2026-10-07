import tempfile
import unittest
from pathlib import Path
import numpy as np

try:
    from scripts.supervised_aoi import Session
except ImportError:
    Session = None


POLYS = {'screen': [[[0,0],[30,0],[30,30],[0,30]],
                    [[30,0],[60,0],[60,30],[30,30]],
                    [[60,0],[99,0],[99,30],[60,30]]],
         'tablet': [[[40,40],[70,40],[70,59],[40,59]]]}


class Source:
    times = [i / 10 for i in range(80)]
    size = (100,60)
    fps = 10
    identity = 'fixture'
    def frame(self, i): return np.full((60,100,3),i,np.uint8)


class Tracker:
    fail = False
    def __init__(self, frame, polygons, visible):
        self.polygons = polygons
        self.visible = visible
    def update(self, frame):
        return self.polygons, ['平板失跟'] if self.fail else []


class SessionTests(unittest.TestCase):
    def test_continue_defers_tracker_setup_until_first_new_frame(self):
        self.seed();calls=[]
        def factory(*args):
            calls.append(1)
            return Tracker(*args)
        self.s.factory=factory
        token=self.s.start(require_review=False)
        self.assertEqual(calls,[])
        self.s.step(token)
        self.assertEqual(calls,[1])

    def setUp(self):
        self.assertIsNotNone(Session, 'supervised session is not implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest = [dict(subject_id=25,segment_id='1.1',start_s=.25,end_s=4.,video_path='fixture'),
                         dict(subject_id=25,segment_id='1.2',start_s=5.,end_s=7.,video_path='fixture')]
        self.path = Path(self.tmp.name)
        self.s = Session(self.manifest, self.path, Source(), POLYS, tracker_factory=Tracker, checkpoint_s=1.)
    def tearDown(self):
        if hasattr(self,'s'): self.s.close()
        if hasattr(self,'tmp'): self.tmp.cleanup()
        Tracker.fail=False
    def seed(self):
        self.s.correct(POLYS, {'screen':True,'tablet':True})
    def test_slice_bounds_and_explicit_initialization(self):
        self.assertEqual(self.s.cursor,3)
        with self.assertRaises(ValueError): self.s.start()
        self.seed()
        self.assertLess(self.s.last_index,40)
        self.s.select('1.2')
        self.assertEqual(self.s.cursor,50)
        with self.assertRaises(ValueError): self.s.start()
    def test_pause_rejects_queued_step_and_survives_restart(self):
        self.seed(); token=self.s.start(); self.s.step(token)
        index=self.s.cursor; self.s.pause()
        with self.assertRaises(ValueError): self.s.step(token)
        self.assertEqual(self.s.cursor,index)
        self.s.close()
        self.s=Session(self.manifest,self.path,Source(),POLYS,tracker_factory=Tracker,checkpoint_s=1.)
        self.assertFalse(self.s.running)
        self.assertEqual(self.s.cursor,index)

    def test_new_slice_inherits_previous_last_outline_and_visibility(self):
        self.seed()
        p={'screen':[[[0,25],[30,20],[60,22],[99,28]]], 'tablet':POLYS['tablet']}
        self.s.seek(12);self.s.correct(p,{'screen':True,'tablet':False})
        expected=self.s.get_record(12)['polygons']
        self.s.seek(3)  # Viewing an earlier frame must not replace the last outline.
        self.s.select('1.2');state=self.s.snapshot()
        self.assertEqual(state['polygons'],expected)
        self.assertEqual(state['visible'],{'screen':True,'tablet':False})
        self.assertEqual(state['inherited_from'],'1.1')
        self.assertFalse(state['record_exists'])
        self.assertFalse(state['approved'])
        self.s.close()
        self.s=Session(self.manifest,self.path,Source(),POLYS,tracker_factory=Tracker)
        self.assertEqual(self.s.snapshot()['polygons'],expected)

    def test_saved_slice_uses_own_outline_instead_of_previous_slice(self):
        self.seed();self.s.select('1.2')
        own={'screen':[[[0,40],[30,32],[60,35],[99,38]]], 'tablet':POLYS['tablet']}
        self.s.correct(own,{'screen':True,'tablet':True})
        expected=self.s.snapshot()['polygons']
        self.s.select('1.1');self.seed();self.s.select('1.2')
        self.assertEqual(self.s.snapshot()['polygons'],expected)
        self.assertIsNone(self.s.snapshot()['inherited_from'])

    def test_first_slice_does_not_inherit_from_future_slice(self):
        self.s.select('1.2');self.seed();self.s.select('1.1')
        self.assertEqual(self.s.snapshot()['polygons'],POLYS)
        self.assertIsNone(self.s.snapshot()['inherited_from'])
    def test_loss_stops_and_requires_correction(self):
        self.seed(); token=self.s.start(); Tracker.fail=True; self.s.step(token)
        self.assertFalse(self.s.running)
        self.assertEqual(self.s.mode,'issue')
        with self.assertRaises(ValueError): self.s.start()
        with self.assertRaises(ValueError): self.s.approve()
        Tracker.fail=False; self.s.correct(POLYS,{'screen':True,'tablet':True})
        self.assertTrue(self.s.start())
    def test_checkpoint_needs_human_approval(self):
        self.seed(); token=self.s.start()
        while self.s.running: self.s.step(token)
        self.assertEqual(self.s.mode,'checkpoint')
        with self.assertRaises(ValueError): self.s.start()
        self.assertEqual(self.s.export(True)['frames'],0)
        self.s.approve()
        self.assertGreater(self.s.export(True)['frames'],0)
    def test_correction_keeps_future_anchor_and_invalidates_review(self):
        self.seed(); token=self.s.start()
        for _ in range(4): self.s.step(token)
        self.s.pause(); self.s.correct(POLYS,{'screen':True,'tablet':True})
        anchor=self.s.cursor
        self.s.seek(3); self.s.correct(POLYS,{'screen':True,'tablet':True})
        self.assertIsNotNone(self.s.get_anchor(anchor))
        self.assertIsNone(self.s.get_record(4))
        self.assertIsNotNone(self.s.get_record(anchor))
        self.assertEqual(self.s.export(True)['frames'],0)

    def test_reset_from_cursor_removes_future_anchors_and_recomputes(self):
        self.seed();token=self.s.start()
        for _ in range(4):self.s.step(token)
        self.s.pause();self.s.correct(POLYS,{'screen':True,'tablet':True})
        future=self.s.cursor
        self.s.select('1.2');self.seed();other=self.s.get_record(50)
        self.s.select('1.1');self.s.seek(4)
        earlier=self.s.get_record(3)
        self.s.reset_from_here(POLYS,{'screen':True,'tablet':True})
        self.assertEqual(self.s.cursor,4)
        self.assertEqual(self.s.get_record(3),earlier)
        self.assertIsNone(self.s.get_anchor(future))
        self.assertIsNone(self.s.get_record(future))
        self.assertTrue(list((self.path/'backups').glob('before-reset-*.sqlite3')))
        token=self.s.continue_simple();self.s.step(token)
        self.assertEqual(self.s.cursor,5)
        self.assertIsNotNone(self.s.get_record(5))
        self.s.select('1.2');self.assertEqual(self.s.get_record(50),other)

    def test_invalid_reset_does_not_delete_annotations(self):
        self.seed();before=self.s.get_record(3)
        with self.assertRaises(ValueError):self.s.reset_from_here({}, {'screen':True,'tablet':True})
        self.assertEqual(self.s.get_record(3),before)
    def test_seek_uncomputed_frame_cannot_skip_unreviewed_gap(self):
        self.s.seek(20); self.seed()
        with self.assertRaises(ValueError): self.s.approve()

    def test_seek_same_frame_does_not_bypass_two_second_trial_review(self):
        self.s.checkpoint_s=10.
        self.seed();token=self.s.start()
        while self.s.running:self.s.step(token)
        self.assertEqual(self.s.mode,'checkpoint')
        self.s.seek(self.s.cursor)
        with self.assertRaises(ValueError):self.s.start()

    def test_restarting_after_invisible_tablet_does_not_expand_saved_geometry(self):
        self.s.correct(POLYS,{'screen':True,'tablet':False})
        token=self.s.start();self.s.step(token);self.s.pause()
        before=self.s.get_record(self.s.cursor)['polygons']
        self.s.start()
        self.assertEqual(self.s.get_record(self.s.cursor)['polygons'],before)

    def test_rejects_crossed_polygon_without_erasing_previous_record(self):
        self.seed();before=self.s.get_record(3)
        import copy
        p=copy.deepcopy(POLYS);p['tablet'][0][1],p['tablet'][0][2]=p['tablet'][0][2],p['tablet'][0][1]
        with self.assertRaises(ValueError):self.s.correct(p,{'screen':True,'tablet':True})
        self.assertEqual(self.s.get_record(3),before)

    def test_continue_from_reviewed_history_reuses_records(self):
        self.seed();token=self.s.start()
        while self.s.running:self.s.step(token)
        self.s.approve()
        before=self.s.export(True)['frames']
        self.s.seek(self.s.first_index);token=self.s.start();self.s.step(token);self.s.pause()
        self.assertEqual(self.s.export(True)['frames'],before)

    def test_real_tracker_on_featureless_frames_pauses(self):
        from scripts.supervised_aoi import Tracker as RealTracker
        self.s.factory=RealTracker
        self.seed();token=self.s.start();self.s.step(token)
        self.assertTrue(self.s.running, 'A single low-feature frame is only a warning')
        while self.s.running:self.s.step(token)
        self.assertFalse(self.s.running)
        self.assertEqual(self.s.mode,'issue')
        self.assertTrue(self.s.get_record(self.s.cursor)['problems'])

    def test_three_screen_points_define_upward_coverage(self):
        from scripts.supervised_aoi import coverage_for
        p={'screen':[[[0,30],[50,20],[99,30]]],'tablet':POLYS['tablet']}
        self.s.correct(p,{'screen':True,'tablet':True})
        self.assertEqual(len(self.s.get_record(3)['polygons']['screen'][0]),3)
        coverage=coverage_for(p['screen'],Source.size)
        self.assertEqual(coverage[0][0],[0.,0.])
        self.assertIn([50.,20.],coverage[0])

    def test_four_screen_points_preserve_all_three_lower_edges(self):
        from scripts.supervised_aoi import coverage_for, four_points, tracker_screens
        points=[[0,35],[30,20],[60,25],[99,15]]
        p={'screen':[points],'tablet':POLYS['tablet']}
        self.s.correct(p,{'screen':True,'tablet':True})
        self.assertEqual(self.s.snapshot()['screen_points'],points)
        coverage=coverage_for([points],Source.size)[0]
        self.assertIn([30.,20.],coverage)
        self.assertIn([60.,25.],coverage)
        self.assertEqual(len(tracker_screens(points,Source.size)),3)
        self.assertEqual(four_points([points],Source.size),points)

    def test_simple_continue_saves_edits_without_approving_candidates(self):
        p={'screen':[[[0,30],[50,20],[99,30]]],'tablet':POLYS['tablet']}
        token=self.s.continue_simple(p,{'screen':True,'tablet':True})
        self.s.step(token);self.s.pause()
        self.assertEqual(self.s.export(True)['frames'],0)
        self.assertIsNotNone(self.s.get_anchor(3))

    def test_outer_screen_points_snap_without_changing_lower_edge(self):
        from scripts.supervised_aoi import four_points, coverage_for
        p=[[10,30],[30,20],[60,25],[90,40]]
        locked=four_points([p],Source.size)
        self.assertEqual(locked[0],[0.,35.])
        self.assertEqual(locked[-1],[99.,44.5])
        self.assertEqual(locked[1:3],p[1:3])
        self.assertEqual(coverage_for([p],Source.size)[0][2],coverage_for([locked],Source.size)[0][2])
        self.s.correct({'screen':[p],'tablet':POLYS['tablet']},{'screen':True,'tablet':True})
        saved=self.s.get_record(3)['polygons']['screen'][0]
        self.assertEqual(saved[0][0],0)
        self.assertEqual(saved[-1][0],99)


if __name__ == '__main__': unittest.main()
