import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scripts.supervised_aoi import Session
from scripts.aoi_horizontal import partition_polygons,center_height,center_slope
from scripts.aoi_range_review import prepare_review,generate_review,commit_review,interpolate_shape
from scripts.aoi_batch import fill_next
from scripts.aoi_workflow_api import ReviewWorker
from tests.test_supervised_aoi import Source,Tracker,POLYS

class KeyframeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
  self.s=Session([dict(subject_id=25,segment_id='1.1',start_s=0,end_s=2,video_path='fixture'),dict(subject_id=25,segment_id='1.2',start_s=3,end_s=4,video_path='fixture')],self.path,Source(),POLYS,Tracker)
 def tearDown(self):self.s.close();self.tmp.cleanup()
 def key(self,y=20,slope=0,gap=10):return dict(polygons=partition_polygons(self.s.source.size,y,gap,slope),visible=dict(screen=True,tablet=True))
 def draft(self,a=None,b=None,start=2,end=12):return prepare_review(self.s,start,end,a or self.key(),b or self.key(30,.6,20),method='keyframes')
 def test_line_angle_height_and_gap_use_timestamps_without_decoding(self):
  draft=self.draft();original=list(self.s.source.times)
  self.s.source.times=list(original);self.s.source.times[7]=.4
  with patch.object(self.s.source,'frame',side_effect=AssertionError('must not decode')),patch('scripts.aoi_range_review.Tracker',side_effect=AssertionError('must not track')):
   generate_review(self.s,draft)
  p=draft['records'][7]['polygons'];f=.2
  self.assertAlmostEqual(center_height(p,self.s.source.size),22,places=4)
  self.assertAlmostEqual(center_slope(p,self.s.source.size),np.tan(f*np.arctan(.6)),places=5)
  self.assertAlmostEqual(p['partition']['gap_px'],12)
  self.assertEqual(draft['records'][2]['polygons'],draft['keys'][2]['polygons'])
  self.assertEqual(draft['records'][12]['polygons'],draft['keys'][12]['polygons'])
  self.assertFalse(draft['records'][7]['problems'])
 def test_similarity_rotation_scale_translation_does_not_shrink_rotating_shape(self):
  a=np.array([[-2,-1],[2,-1],[2,1],[-2,1]],dtype=float)
  angle=np.deg2rad(90);r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  b=a@r.T*3+[20,30];mid=np.asarray(interpolate_shape(a,b,.5))
  angle/=2;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  np.testing.assert_allclose(mid,a@r.T*2+[10,15],atol=1e-8)
  np.testing.assert_allclose(interpolate_shape(a,b,1),b,atol=1e-8)
 def test_piecewise_internal_anchor_is_exact_and_only_interval_is_changed(self):
  while fill_next(self.s):pass
  self.s.select('1.1');mid=self.key(15,-.2,7)
  self.s.db.execute('insert into anchors values (?,?,?)',('1.1',7,json.dumps(mid)));self.s.db.commit()
  old=self.s.db.execute('select * from records order by segment,frame').fetchall()
  anchor=self.s.db.execute('select * from anchors where frame=7').fetchone()
  draft=self.draft();generate_review(self.s,draft)
  self.assertEqual(old,self.s.db.execute('select * from records order by segment,frame').fetchall())
  self.assertEqual(draft['records'][7]['polygons'],draft['keys'][7]['polygons'])
  commit_review(self.s,draft)
  new=self.s.db.execute('select * from records order by segment,frame').fetchall()
  outside=lambda rows:[r for r in rows if r[0]!='1.1' or not 2<=r[1]<=12]
  self.assertEqual(outside(old),outside(new));self.assertEqual(anchor,self.s.db.execute('select * from anchors where frame=7').fetchone())
  self.assertEqual(self.s.cursor,12);self.assertEqual(self.s.mode,'paused')
  self.assertEqual(self.s.get_record(6)['tracking']['method'],'manual_keyframe_interpolation')
  self.assertFalse(self.s.get_record(6)['approved']);self.assertTrue(list((self.path/'backups').glob('*.sqlite3')))
  # Batch filling cannot overwrite the repaired records.
  while fill_next(self.s):pass
  self.assertEqual(new,self.s.db.execute('select * from records order by segment,frame').fetchall())
 def test_visibility_change_stays_flagged(self):
  tail=self.key();tail['visible']['tablet']=False;d=self.draft(b=tail);generate_review(self.s,d)
  self.assertIn('可见性变化',d['records'][6]['problems'][0]);self.assertGreater(d['uncertain'],0)
 def test_cancel_and_conflict_never_partially_commit(self):
  d=self.draft()
  with self.assertRaises(InterruptedError):generate_review(self.s,d,cancel=lambda:True)
  with self.assertRaises(ValueError):commit_review(self.s,d)
  self.assertEqual(self.s.db.execute('select count(*) from records').fetchone()[0],0)
  d=self.draft();generate_review(self.s,d);self.s._put(8,self.key()['polygons'],self.key()['visible'],[]);self.s.db.commit()
  old=self.s.db.execute('select * from records').fetchall()
  with self.assertRaisesRegex(ValueError,'变化'):commit_review(self.s,d)
  self.assertEqual(old,self.s.db.execute('select * from records').fetchall())
 def test_worker_default_is_keyframes_and_does_not_close_live_decoder(self):
  server=SimpleNamespace(session=self.s);worker=ReviewWorker(server)
  self.s.source.cap=object();self.s.source.close=lambda:self.fail('live decoder closed')
  with patch('scripts.aoi_workflow_api.cv2.VideoCapture',side_effect=AssertionError('must not open decoder')):
   worker.start(dict(start=2,end=12,head=self.key(),tail=self.key(30)))
   worker.thread.join(3)
  self.assertEqual(worker.draft['method'],'keyframes');self.assertEqual(worker.draft['status'],'ready')

if __name__=='__main__':unittest.main()
