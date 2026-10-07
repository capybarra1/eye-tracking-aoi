import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from scripts import aoi_line_cache as cache

class LineCacheTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'line.sqlite3';self.gray=np.random.default_rng(12).integers(0,256,(270,480),dtype=np.uint8)
  self.writer=cache.LineWriter(self.path,max_bytes=8_000_000,min_free_bytes=0)
 def tearDown(self):self.writer.close();cache.configure_cache(None);self.tmp.cleanup()
 def test_actual_scene_detector_matches_direct_masked_and_unmasked_features(self):
  self.writer.prepare(self.gray,owner='video');cache.configure_cache(self.path)
  from scripts.aoi_scene_anchor import SceneAnchor
  wrapped=SceneAnchor().detector;original=cv2.SIFT_create(nfeatures=600,contrastThreshold=.015)
  attrs=lambda keys:[(*k.pt,k.size,k.angle,k.response,k.octave,k.class_id) for k in keys]
  for mask in [None,np.zeros_like(self.gray),np.random.default_rng(9).integers(0,2,self.gray.shape,dtype=np.uint8)*255]:
   a,b=original.detectAndCompute(self.gray,mask);c,d=wrapped.detectAndCompute(self.gray,mask)
   self.assertEqual(attrs(a),attrs(c));self.assertEqual(b is None,d is None)
   if b is not None:np.testing.assert_array_equal(b,d)
  self.assertEqual(wrapped.hits,3)
 def test_resume_does_not_compute_twice(self):
  self.assertTrue(self.writer.prepare(self.gray,owner='video'))
  with patch.object(self.writer,'detector',None):self.assertFalse(self.writer.prepare(self.gray,owner='video'))
 def test_corruption_or_changed_image_falls_back(self):
  self.writer.prepare(self.gray,owner='video');cache.configure_cache(self.path);detector=cache.scene_sift()
  self.writer.db.execute("UPDATE features SET data=x'ff'");self.writer.db.commit()
  a,b=detector.detectAndCompute(self.gray,None);x,y=cv2.SIFT_create(nfeatures=600,contrastThreshold=.015).detectAndCompute(self.gray,None)
  np.testing.assert_array_equal(b,y);self.assertEqual(detector.hits,0)
  changed=self.gray.copy();changed[0,0]^=255;detector.detectAndCompute(changed,None);self.assertEqual(detector.hits,0)
 def test_space_reasons_are_distinct_and_no_partial_write(self):
  self.writer.max_bytes=1
  with self.assertRaisesRegex(OSError,'缓存预算'):self.writer.prepare(self.gray,owner='video')
  self.assertEqual(self.writer.db.execute('SELECT count(*) FROM features').fetchone()[0],0)
  self.writer.max_bytes=8_000_000
  with patch('scripts.aoi_line_cache.shutil.disk_usage') as usage:
   usage.return_value.free=0
   with self.assertRaisesRegex(OSError,'硬盘剩余'):self.writer.prepare(self.gray,owner='video')
 def test_retiring_cache_is_limited_to_named_owners(self):
  self.writer.prepare(self.gray,owner='done');other=self.gray.copy();other[0,0]^=255;self.writer.prepare(other,owner='active')
  self.writer.retire_owners(['done']);self.assertEqual(self.writer.db.execute('SELECT owner FROM features').fetchall(),[('active',)])

if __name__=='__main__':unittest.main()
