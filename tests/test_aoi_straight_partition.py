import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from scripts.aoi_horizontal import partition_polygons, HorizontalTracker, center_height, center_slope
from scripts.supervised_aoi import effective_aois, validate_polygons, Tracker, Session
from scripts.aoi_range_review import prepare_review, generate_review, commit_review
from tests.test_supervised_aoi import Source, POLYS


def scene(y=300,slope=.1):
    h,w=540,960;yy,xx=np.ogrid[:h,:w];line=y+slope*(xx-(w-1)/2)
    a=np.where(yy<line-14,100,np.where(yy<line,15,170)).astype('uint8')
    return cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)


class StraightPartitionTests(unittest.TestCase):
    def test_tilt_gap_and_clipping(self):
        p=partition_polygons((960,540),300,24,.3)
        regions=effective_aois(p,dict(screen=True,tablet=True),(960,540))
        for x in (1,200,480,950):
            y=300+.3*(x-479.5)
            def inside(key,dy):return any(cv2.pointPolygonTest(np.float32(poly),(x,y+dy),False)>0 for poly in regions[key])
            self.assertTrue(inside('screen',-2));self.assertTrue(inside('tablet',30))
            self.assertFalse(inside('screen',12));self.assertFalse(inside('tablet',12))
        self.assertAlmostEqual(center_slope(validate_polygons(p,dict(screen=True,tablet=True),(960,540)),(960,540)),.3,places=5)

    def test_no_heavy_tracker_and_follows_rotation(self):
        p=partition_polygons((960,540),300,24,.1)
        with patch('scripts.supervised_aoi.PilotState',side_effect=AssertionError('heavy pipeline')):
            t=Tracker(scene(),p,dict(screen=True,tablet=True))
            self.assertIsInstance(t,HorizontalTracker)
            for y,slope in ((304,.11),(310,.12),(316,.14)):
                result,problems=t.update(scene(y,slope))
                if problems:result,problems=t.update(scene(y,slope))
                self.assertFalse(problems)
                self.assertAlmostEqual(center_height(result,(960,540)),y,delta=3)
                self.assertAlmostEqual(center_slope(result,(960,540)),slope,delta=.025)
            held=copy.deepcopy(result)
            result,problems=t.update(np.zeros((540,960,3),np.uint8))
            self.assertTrue(problems);self.assertEqual(result,held)

    def test_custom_gap_survives_resume(self):
        with tempfile.TemporaryDirectory() as d:
            s=Session([dict(subject_id=25,segment_id='1.1',start_s=0,end_s=7,video_path='fixture')],Path(d),Source(),POLYS)
            s.set_partition(True)
            p=partition_polygons(Source.size,20,12,.1);s.correct(p,dict(screen=True,tablet=True))
            tracker=s._make_tracker(s.cursor,s.get_record(s.cursor))
            self.assertEqual(tracker.gap,12)
            token=s.start(require_review=False);s.step(token)
            self.assertEqual(s.get_record(s.cursor)['polygons']['partition']['gap_px'],12)
            s.close()

    def test_detector_rounding_at_crop_border(self):
        p=partition_polygons((960,540),300,24,0)
        t=HorizontalTracker(scene(300,0),p,dict(screen=True,tablet=True))
        class EdgeDetector:
            def detect(self,crop):return (np.float32([[[0,149,145.5,149]]]),None,None,None)
        t.lsd=EdgeDetector()
        result,_=t.update(scene(300,0))
        self.assertTrue(result['partition'])

    def test_mode_switch_preserves_other_frames_and_archive_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            s=Session([dict(subject_id=25,segment_id='1.1',start_s=0,end_s=7,video_path='fixture')],Path(d),Source(),POLYS)
            s.correct(POLYS,dict(screen=True,tablet=True))
            for i in (1,2,3):s._put(i,POLYS,dict(screen=True,tablet=True),[])
            s._save();before=s.db.execute('SELECT * FROM records WHERE frame>0').fetchall()
            s.set_partition(True)
            self.assertEqual(before,s.db.execute('SELECT * FROM records WHERE frame>0').fetchall())
            self.assertTrue(s.get_record(0)['polygons']['partition'])
            self.assertTrue(list((Path(d)/'backups').glob('*.sqlite3')))
            token=s.start(require_review=False)
            for _ in range(4):s.step(token)
            self.assertTrue(s.get_record(4)['polygons']['partition'])
            self.assertEqual(s.mode,'running')
            s.close()

    def test_local_review_keeps_mode_and_outside_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            s=Session([dict(subject_id=25,segment_id='1.1',start_s=0,end_s=7,video_path='fixture')],Path(d),Source(),POLYS)
            s.set_partition(True)
            head=dict(polygons=partition_polygons(Source.size,30,8,.1),visible=dict(screen=True,tablet=True))
            tail=dict(polygons=partition_polygons(Source.size,35,8,.2),visible=dict(screen=True,tablet=True))
            before=s.get_record(0)
            draft=prepare_review(s,2,4,head,tail);generate_review(s,draft);commit_review(s,draft)
            self.assertEqual(before,s.get_record(0))
            self.assertTrue(s.get_record(3)['polygons']['partition'])
            self.assertAlmostEqual(center_slope(s.get_record(4)['polygons'],Source.size),.2,places=5)
            s.close()
