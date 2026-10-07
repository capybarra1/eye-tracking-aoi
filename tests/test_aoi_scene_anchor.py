import unittest
import cv2,numpy as np
from scripts.aoi_scene_anchor import SceneAnchor

class SceneAnchorTests(unittest.TestCase):
 def image(self):
  rng=np.random.default_rng(172);im=np.full((270,480),24,np.uint8)
  for _ in range(170):
   x,y=rng.integers([15,65],[465,235]);radius=int(rng.integers(2,6));cv2.circle(im,(int(x),int(y)),radius,int(rng.integers(40,245)),-1)
  return im
 def test_matches_static_scene_when_boundary_leaves_centre_view(self):
  im=self.image();anchor=SceneAnchor();anchor.add(im,90,-.1)
  A=cv2.getRotationMatrix2D((240,135),4,1);A[:,2]+=[7,-85];changed=cv2.warpAffine(im,A,(480,270))
  c=anchor.candidate(changed);self.assertIsNotNone(c)
  ends=np.float32([[200,94,1],[280,86,1]])@A.T;s=np.diff(ends[:,1])[0]/np.diff(ends[:,0])[0];y=ends[0,1]+(240-ends[0,0])*s
  self.assertAlmostEqual(c['y'],y,delta=1);self.assertAlmostEqual(c['slope'],s,delta=.02)
 def test_black_or_unrelated_image_does_not_match(self):
  im=self.image();anchor=SceneAnchor();anchor.add(im,90,-.1)
  self.assertIsNone(anchor.candidate(np.zeros_like(im)))
  self.assertIsNone(anchor.candidate(np.random.default_rng(99).integers(0,255,im.shape,dtype=np.uint8)))
 def test_descriptors_and_manual_geometry_remain_immutable(self):
  im=self.image();anchor=SceneAnchor();anchor.add(im,90,-.1);before=anchor.refs[0]['descriptors'].copy()
  for _ in range(4):anchor.candidate(im)
  np.testing.assert_array_equal(anchor.refs[0]['descriptors'],before);self.assertEqual(anchor.refs[0]['y'],90)

class SessionSceneReferencesTests(unittest.TestCase):
 def test_generated_geometry_is_not_used_as_absolute_reference(self):
  import tempfile
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import patch
  from scripts.supervised_aoi import Session
  from scripts.aoi_horizontal import partition_polygons
  from tests.test_supervised_aoi import Source,Tracker
  with tempfile.TemporaryDirectory() as folder:
   poly=partition_polygons((100,60),25,12);visible=dict(screen=True,tablet=True)
   s=Session([dict(subject_id=1,segment_id='1.1',start_s=0,end_s=7.9,video_path='fixture')],Path(folder),Source(),poly,Tracker)
   s.correct(poly,visible);s.seek(4);s._put(4,poly,visible,[])
   def factory(source,kind,index,polygons,visible):
    return SimpleNamespace(scene=SimpleNamespace(refs=[index]),bank=[],tablet_reference=None,upper=SimpleNamespace(reference=None))
   with patch('scripts.supervised_aoi.make_tracker',side_effect=factory):
    t=s._make_tracker(4,s.get_record(4));self.assertEqual(t.scene.refs,[0])
   s.close()

if __name__=='__main__':unittest.main()
