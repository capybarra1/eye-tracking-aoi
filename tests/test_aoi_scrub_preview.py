import json
import unittest
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError
from unittest.mock import patch
from tests.test_aoi_saving import SavingHTTPTests
from scripts.aoi_horizontal import partition_polygons


class ScrubPreviewTests(unittest.TestCase):
    setUp=SavingHTTPTests.setUp
    tearDown=SavingHTTPTests.tearDown
    action=SavingHTTPTests.action

    def preview(self,index,**overrides):
        args=dict(project=self.manager.current,segment=self.s.segment,revision=self.server.revision,index=index);args.update(overrides)
        try:
            with urlopen(self.url+'/api/preview?'+urlencode(args)) as r:return r.status,json.load(r)
        except HTTPError as e:return e.code,json.load(e)

    def test_exact_saved_geometry_without_writes_or_cursor_change(self):
        index=self.s.cursor+1
        p=partition_polygons(self.s.source.size,25,12,.1)
        self.s._put(index,p,dict(screen=True,tablet=True),['屏幕短暂失跟']);self.s._save()
        before=(self.s.cursor,self.server.revision,list(self.s.db.iterdump()))
        code,result=self.preview(index)
        self.assertEqual(code,200);self.assertEqual(result['cursor'],index)
        self.assertEqual(result['polygons'],p);self.assertEqual(result['time_s'],self.s.source.times[index])
        self.assertTrue(result['image'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(before,(self.s.cursor,self.server.revision,list(self.s.db.iterdump())))

    def test_unannotated_has_no_borrowed_geometry_and_frames_are_cached(self):
        index=self.s.cursor+1
        with patch.object(self.s.source,'frame',wraps=self.s.source.frame) as read:
            for _ in range(2):
                code,result=self.preview(index);self.assertEqual(code,200)
                self.assertFalse(result['record_exists']);self.assertIsNone(result['polygons']);self.assertIsNone(result['screen_points'])
            self.assertEqual(read.call_count,1)

    def test_reject_stale_project_slice_revision_running_and_outside_slice(self):
        for args in [dict(project='other'),dict(segment='other'),dict(revision=-1)]:
            self.assertEqual(self.preview(self.s.cursor,**args)[0],409)
        self.assertEqual(self.preview(self.s.last_index+1)[0],400)
        self.s.start(require_review=False)
        self.assertEqual(self.preview(self.s.cursor)[0],400)
        self.assertTrue(self.s.running)

    def test_cache_does_not_cache_old_annotation(self):
        i=self.s.cursor;self.preview(i)
        p=partition_polygons(self.s.source.size,28,8,.2)
        self.s.correct(p,dict(screen=True,tablet=True))
        self.assertEqual(self.preview(i)[1]['polygons'],self.s.get_record(i)['polygons'])
