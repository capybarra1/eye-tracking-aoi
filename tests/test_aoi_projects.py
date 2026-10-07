import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.aoi_projects import Projects, make_package, validate_package, validate_manifest, SEGMENTS
from scripts.supervised_aoi import Session
from tests.test_supervised_aoi import Source, Tracker, POLYS


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.video=self.root/'scenevideo.mp4';self.video.write_bytes(b'video-fixture')
        self.pts=self.root/'pts.json';self.pts.write_text(json.dumps(Source.times))
        self.source=Source();self.source.path=self.video;self.source.close=lambda:None
        self.rows=[dict(subject_id=25,segment_id=s,start_s=i*.6,end_s=i*.6+.5,video_path=str(self.video)) for i,s in enumerate(SEGMENTS)]
        timing=self.root/'synthetic-timing';timing.mkdir()
        (timing/'eligible_subject_inventory.json').write_text(json.dumps([dict(subject_id=25,eligible=True,image_count=0)]))
        (timing/'demo_segments.json').write_text(json.dumps(self.rows))
        timing_patch=patch('scripts.aoi_projects.TIMING',timing)
        timing_patch.start();self.addCleanup(timing_patch.stop)
        self.s=Session(self.rows,self.root/'original',self.source,POLYS,Tracker)
        self.s.correct(POLYS,dict(screen=True,tablet=True));self.s.continue_simple();self.s.step(self.s.token);self.s.pause()
        self.manager=Projects(self.s,self.pts,self.root/'projects')
    def tearDown(self):self.s.close();self.tmp.cleanup()

    def test_round_trip_preserves_geometry_and_separates_workspace(self):
        package=make_package(self.s,self.manager.digest(self.video))
        original=self.s.db.execute('select segment,frame,payload,approved from records order by frame').fetchall()
        with patch('scripts.aoi_projects.VideoSource',return_value=self.source):new=self.manager.import_review(package)
        try:
            actual=new.db.execute('select segment,frame,payload,approved from records order by frame').fetchall()
            self.assertEqual([(a,b,json.loads(c),d) for a,b,c,d in actual],[(a,b,json.loads(c),d) for a,b,c,d in original])
            self.assertNotEqual(new.folder,self.s.folder)
            self.assertFalse(new.running)
            self.assertEqual(len(new.snapshot()['slices']),12)
            self.assertEqual(len(new.db.execute('select * from anchors').fetchall()),1)
        finally:new.close()

    def test_rejects_wrong_video_and_modified_time_axis(self):
        package=make_package(self.s,'fingerprint')
        with self.assertRaises(ValueError):validate_package(package,self.source,'wrong',self.rows)
        package['timestamps']=list(package['timestamps']);package['timestamps'][1]+=.01
        with self.assertRaises(ValueError):validate_package(package,self.source,'fingerprint',self.rows)

    def test_rejects_frame_outside_its_slice_and_duplicate(self):
        package=make_package(self.s,'fingerprint');package['records'][0]['segment']='6.2'
        with self.assertRaises(ValueError):validate_package(package,self.source,'fingerprint',self.rows)
        package=make_package(self.s,'fingerprint');package['records'].append(copy.deepcopy(package['records'][0]))
        with self.assertRaises(ValueError):validate_package(package,self.source,'fingerprint',self.rows)

    def test_exact_twelve_slices_and_no_overlap(self):
        with self.assertRaises(ValueError):validate_manifest(self.rows[:-1],25,self.video)
        rows=copy.deepcopy(self.rows);rows[1]['start_s']=0
        with self.assertRaises(ValueError):validate_manifest(rows,25,self.video)
        self.assertEqual(len(validate_manifest(self.rows[:2],25,self.video,partial=True)),2)
        with self.assertRaises(ValueError):validate_manifest([self.rows[0],self.rows[0]],25,self.video,partial=True)
