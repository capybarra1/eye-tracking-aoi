import copy,json,tempfile,unittest
from pathlib import Path
from tests.test_supervised_aoi import Source,Tracker,POLYS
from scripts.supervised_aoi import Session
from scripts.aoi_projects import make_package,validate_package

class TimingTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
  self.rows=[dict(subject_id=25,segment_id='1.1',start_s=1.,end_s=4.,video_path='fixture'),dict(subject_id=25,segment_id='1.2',start_s=5.,end_s=7.,video_path='fixture')]
  self.source=Source();self.source.path=Path('fixture');self.s=Session(copy.deepcopy(self.rows),self.path,self.source,POLYS,tracker_factory=Tracker)
  for index in (10,20,39):self.s.seek(index);self.s.correct(POLYS,dict(screen=True,tablet=True))
 def tearDown(self):self.s.close();self.tmp.cleanup()
 def adjust(self,start,end):
  from scripts.aoi_slice_timing import adjust_timing
  return adjust_timing(self.s,start,end)
 def test_extend_preserves_records_and_starts_at_new_unmarked_head(self):
  before=self.s.db.execute('select * from records').fetchall();self.adjust(.5,4.5)
  self.assertEqual((self.s.first_index,self.s.last_index,self.s.cursor),(5,44,5));self.assertFalse(self.s.snapshot()['record_exists'])
  self.assertEqual(before,self.s.db.execute('select * from records').fetchall())
  self.assertEqual(self.s.snapshot()['slices'][0]['computed'],3)
 def test_shrink_excludes_records_from_export_but_restoring_recovers_them(self):
  self.adjust(2.,3.);self.assertEqual(self.s.snapshot()['slices'][0]['computed'],1)
  package=make_package(self.s,'digest');self.assertEqual([r['frame'] for r in package['records']],[20])
  validate_package(package,self.source,'digest',self.s.manifest)
  self.adjust(1.,4.);self.assertEqual(self.s.snapshot()['slices'][0]['computed'],3)
 def test_restart_uses_override_and_keeps_base_identity(self):
  identity=self.s._get('identity');self.adjust(.5,4.5);self.s.close()
  self.s=Session(copy.deepcopy(self.rows),self.path,self.source,POLYS,tracker_factory=Tracker)
  self.assertEqual(self.s.first_index,5);self.assertEqual(self.s.last_index,44);self.assertEqual(self.s._get('identity'),identity)
  self.assertEqual(self.s.base_manifest,self.rows)
 def test_invalid_or_overlap_is_rejected_without_mutation(self):
  before=self.s.snapshot()
  for a,b in [(3,2),(-1,3),(1,5.2),(float('nan'),3),(1,99)]:
   with self.assertRaises(ValueError):self.adjust(a,b)
  self.assertEqual(self.s.snapshot(),before)
 def test_other_slice_and_anchors_untouched_with_backup(self):
  self.s.select('1.2');self.s.correct(POLYS,dict(screen=True,tablet=True));self.s.select('1.1')
  anchors=self.s.db.execute('select * from anchors').fetchall();self.adjust(1.2,4.5)
  self.assertEqual(anchors,self.s.db.execute('select * from anchors').fetchall());self.assertEqual(self.s.slices['1.2']['first'],50)
  self.assertTrue(list((self.path/'backups').glob('before-timing-*.sqlite3')))
 def test_directory_uses_updated_range(self):
  from scripts.aoi_progress import project_progress
  (self.path/'manifest.json').write_text(json.dumps(self.rows));(self.path/'times.json').write_text(json.dumps(self.source.times))
  p=dict(folder=str(self.path),manifest=str(self.path/'manifest.json'),timestamps=str(self.path/'times.json'))
  self.adjust(2,3);row=project_progress(p)['segments'][0];self.assertEqual((row['total'],row['computed']),(10,1))
 def test_batch_gap_search_ignores_archived_outside_frames(self):
  from scripts.aoi_batch import next_gap
  self.adjust(2.,3.)
  self.assertEqual(next_gap(self.s),('1.1',21))
  for index in range(21,30):self.s._put(index,POLYS,dict(screen=True,tablet=True),[])
  self.assertEqual(next_gap(self.s),('1.2',50))
 def test_readonly_preview_does_not_seek_or_write(self):
  from scripts.aoi_slice_timing import timing_preview
  before=self.s.snapshot();r=timing_preview(self.s,2);self.assertEqual(r['index'],2);self.assertTrue(r['image'].startswith('data:image/jpeg'))
  self.assertEqual(before,self.s.snapshot())

if __name__=='__main__':unittest.main()
