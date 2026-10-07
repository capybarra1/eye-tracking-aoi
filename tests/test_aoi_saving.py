"""Persistence contracts: consistent pairs, interrupted work, and actual HTTP actions."""
import copy
import csv
import io
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import unquote
from urllib.request import Request, urlopen

from scripts.aoi_projects import validate_package
from scripts.supervised_aoi_server import Server
from tests import test_aoi_projects as fixtures
from tests.test_supervised_aoi import POLYS, Tracker


class SavingTests(unittest.TestCase):
    setUp=fixtures.ProjectTests.setUp
    tearDown=fixtures.ProjectTests.tearDown
    def read_pair(self):
        saved=self.manager.saved_annotations(self.s)
        folder=self.s.folder/'exports'
        archive=json.loads((folder/saved['archive_name']).read_text())
        with (folder/saved['csv_name']).open(encoding='utf-8-sig',newline='') as f:table=list(csv.DictReader(f))
        self.assertEqual(len(table),len(archive['records']))
        self.assertEqual(saved['frames'],len(table))
        self.assertTrue(all(r['save_id']==archive['save_id']==saved['save_id'] for r in table))
        validate_package(archive,self.source,self.manager.digest(self.video),self.rows)
        return saved,archive,table

    def test_pair_named_for_subject_and_includes_all_slices_and_pending_records(self):
        self.s.select('1.2');self.s.correct(POLYS,dict(screen=True,tablet=True))
        self.manager.save_annotations(self.s)
        saved,archive,table=self.read_pair()
        self.assertTrue(saved['archive_name'].startswith('25号_'))
        self.assertTrue(saved['archive_name'].endswith('_标注存档.aoi.json'))
        self.assertEqual({r['segment_id'] for r in table},{'1.1','1.2'})
        self.assertTrue(all(r['approved']=='False' for r in table))
        self.assertEqual(len(archive['anchors']),2)

    def test_correction_updates_same_pair_and_round_trips_latest_geometry(self):
        before=self.manager.save_annotations(self.s)
        edited=copy.deepcopy(POLYS);edited['tablet'][0]=[[42,40],[72,40],[72,59],[42,59]]
        self.s.correct(edited,dict(screen=True,tablet=True))
        self.manager.save_annotations(self.s,'手工修正')
        saved,archive,table=self.read_pair()
        self.assertEqual(before['archive_name'],saved['archive_name'])
        self.assertNotEqual(before['save_id'],saved['save_id'])
        self.assertEqual(archive['records'][-1]['payload']['polygons']['tablet'],edited['tablet'])
        self.assertEqual(json.loads(table[-1]['aois_json'])['tablet'],edited['tablet'])
        with patch('scripts.aoi_projects.VideoSource',return_value=self.source):new=self.manager.import_review(archive)
        try:self.assertEqual(new.get_record(new.cursor)['polygons']['tablet'],edited['tablet'])
        finally:new.close()

    def test_publish_failure_keeps_previous_pair_and_saved_time(self):
        saved=self.manager.save_annotations(self.s)
        original=Path.replace
        def fail_csv(path,target):
            if path.name==saved['csv_name'] and path.parent.name.startswith('.saving-'):
                raise OSError('disk full')
            return original(path,target)
        with patch.object(Path,'replace',fail_csv):
            with self.assertRaises(OSError):self.manager.save_annotations(self.s)
        after,_,_=self.read_pair()
        self.assertEqual(saved,after)

    def test_same_subject_different_recordings_have_different_names(self):
        before=self.manager.save_annotations(self.s)
        video=self.root/'second_recording'/'scenevideo.mp4';video.parent.mkdir();video.write_bytes(b'other')
        self.source.path=video
        try:after=self.manager.save_annotations(self.s)
        finally:self.source.path=self.video
        self.assertNotEqual(before['archive_name'],after['archive_name'])


class SavingHTTPTests(unittest.TestCase):
    def setUp(self):
        fixtures.ProjectTests.setUp(self)
        self.server=Server(('127.0.0.1',0),self.s,self.manager)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
        fixtures.ProjectTests.tearDown(self)
    def action(self,action,**data):
        data['revision']=self.server.revision
        req=Request(self.url+'/api/'+action,data=json.dumps(data).encode(),headers={'Content-Type':'application/json','X-AOI-Client':'1'})
        try:
            with urlopen(req) as r:return r.status,json.load(r)
        except HTTPError as e:return e.code,json.load(e)
    def test_pause_and_manual_save_while_running(self):
        code,state=self.action('continue');self.assertEqual(code,200)
        self.action('step',token=state['token'])
        code,state=self.action('pause');self.assertEqual(code,200)
        self.assertFalse(state['running']);self.assertEqual(state['saved']['reason'],'暂停保存')
        self.action('continue')
        code,state=self.action('save');self.assertEqual(code,200)
        self.assertFalse(state['running']);self.assertEqual(state['saved']['reason'],'手动保存')

    def test_reset_http_saves_pair_and_restarts_at_selected_frame(self):
        code,state=self.action('continue');self.assertEqual(code,200)
        self.action('step',token=state['token'])
        target=self.s.cursor
        self.action('step',token=state['token'])
        self.action('pause')
        self.action('seek',index=target)
        code,state=self.action('reset',polygons=POLYS,visible=dict(screen=True,tablet=True))
        self.assertEqual(code,200)
        self.assertEqual(state['cursor'],target)
        self.assertEqual(state['saved']['reason'],'从当前位置重标')
        self.assertIsNone(self.s.get_record(target+1))
        code,state=self.action('continue');self.assertEqual(code,200)
        code,state=self.action('step',token=state['token']);self.assertEqual(code,200)
        self.assertEqual(state['cursor'],target+1)
    def test_slice_end_checkpoint_and_anomaly_save_without_export_click(self):
        code,state=self.action('continue')
        while state['running']:
            code,state=self.action('step',token=state['token'])
            self.assertEqual(code,200)
        self.assertEqual(state['saved']['reason'],'切片结束')
        saved=self.server.saves.wait(self.s,state['saved']['generation'])
        self.assertEqual(saved['frames'],5)
        self.action('select',segment='1.2')
        self.action('correct',polygons=POLYS,visible=dict(screen=True,tablet=True))
        code,state=self.action('continue')
        self.s.next_check=self.s.source.times[self.s.cursor]+.1
        code,state=self.action('step',token=state['token'])
        self.assertEqual(state['saved']['reason'],'检查点暂停')
        code,state=self.action('continue');Tracker.fail=True
        try:code,state=self.action('step',token=state['token'])
        finally:Tracker.fail=False
        self.assertEqual(code,200);self.assertEqual(state['saved']['reason'],'异常暂停')
        self.assertEqual(state['mode'],'issue')
    def test_manual_save_commits_unsent_correction_and_named_downloads(self):
        edited=copy.deepcopy(POLYS);edited['tablet'][0][0][0]+=1
        code,state=self.action('save',polygons=edited,visible=dict(screen=True,tablet=True))
        self.assertEqual(code,200)
        for route,key in [('archive','archive_name'),('table','csv_name')]:
            with urlopen(self.url+'/api/annotations/'+route) as r:
                self.assertIn(state['saved'][key],unquote(r.headers['Content-Disposition']))
                data=r.read()
                if route=='archive':self.assertEqual(json.loads(data)['records'][-1]['payload']['polygons']['tablet'],edited['tablet'])
        with self.assertRaises(HTTPError) as error:urlopen(self.url+'/api/annotations/table?project=wrong')
        self.assertEqual(error.exception.code,409)
    def test_failure_is_visible_and_retry_recovers(self):
        self.action('save')
        with patch.object(self.manager,'save_annotations',side_effect=OSError('disk full')):
            code,state=self.action('pause')
            self.assertEqual(code,200)
            with self.assertRaisesRegex(OSError,'disk full'):
                self.server.saves.wait(self.s,state['saved']['generation'])
        saved=self.server.saves.status(self.s)
        self.assertEqual(saved['status'],'error')
        self.assertIn('disk full',saved['error'])
        code,state=self.action('save')
        self.assertEqual(code,200);self.assertEqual(state['saved']['status'],'saved')
