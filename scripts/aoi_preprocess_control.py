"""Control the independent cache worker without touching annotation state."""
from __future__ import annotations

import fcntl
import json
import shutil
import sqlite3
import time
import subprocess
import sys
import threading
from pathlib import Path


class PreprocessControl:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.current_project=None;self.requested_project=None;self.storage_cache=None;self.storage_time=0.
        self.intent_path=self.root/'outputs/line_preprocess_intent.json'
        try:self.auto_prepare=bool(json.loads(self.intent_path.read_text()).get('enabled'))
        except (OSError,ValueError):self.auto_prepare=False

    def config(self):
        path=self.root/'models/line_cache.json'
        mode='line' if path.exists() else 'legacy'
        config=json.loads((path if mode=='line' else self.root/'models/feature_cache.json').read_text())
        return mode,config

    def database(self) -> Path:
        mode,config = self.config()
        if not config.get('enabled'):
            raise ValueError('尚未启用预处理')
        path = Path(config['database']).expanduser().resolve()
        if not path.parent.is_dir():
            raise ValueError('预处理硬盘未连接')
        return path

    @staticmethod
    def running(folder: Path) -> bool:
        path = folder / 'preprocess.lock'
        if not path.exists():
            return False
        with path.open('r') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
        return False

    def status(self) -> dict:
        with self.lock:
            try:
                database=self.database();folder = database.parent;mode,config=self.config()
                try:
                    progress = json.loads((folder / 'progress.json').read_text())
                except (OSError, ValueError):
                    progress = {}
                result = {'enabled': True, **{k: progress.get(k) for k in
                          ('status', 'subject', 'segment', 'frames_this_run', 'recordings', 'error','segments','completed_jobs','job_start','job_end','position_s')}}
                if self.storage_cache is None or time.monotonic()-self.storage_time>2:
                    allocated=sum(p.stat().st_size for p in [database,Path(str(database)+'-wal')] if p.exists());used=allocated
                    if mode=='line' and database.exists():
                        try:
                            with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=.02) as db:
                                used=(db.execute('PRAGMA page_count').fetchone()[0]-db.execute('PRAGMA freelist_count').fetchone()[0])*db.execute('PRAGMA page_size').fetchone()[0]
                        except sqlite3.Error:pass
                    self.storage_cache=dict(cache_bytes=used,allocated_bytes=allocated,free_gb=round(shutil.disk_usage(folder).free/1024**3,1));self.storage_time=time.monotonic()
                result.update(mode=mode,max_gb=config.get('max_gb',12 if mode=='line' else 48),**self.storage_cache)
                if mode=='line':
                    from scripts.aoi_line_cache import runtime_status
                    result.update(runtime_status())
                if self.running(folder):
                    result['status'] = 'pausing' if (folder / 'STOP').exists() else 'running'
                elif self.process is not None and self.process.poll() is None:
                    result['status'] = 'starting'
                elif self.process is not None and self.process.returncode != 0:
                    result.update(status='error', error='预处理启动失败，请检查硬盘与视频文件后重试')
                elif result['status'] not in ('complete', 'error','space_paused'):
                    result['status'] = 'paused'
                if mode=='line' and self.auto_prepare and self.current_project and progress.get('current_project')!=self.current_project:
                    if result['status']=='running':
                        (folder/'STOP').touch();result['status']='pausing'
                    elif result['status'] not in ('starting','pausing') and self.requested_project!=self.current_project:
                        self._spawn(folder);result['status']='starting'
                return result
            except (OSError, ValueError, KeyError) as exc:
                return {'enabled': False, 'status': 'error', 'error': str(exc)}

    def _intent(self,enabled):
        self.auto_prepare=enabled
        self.intent_path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.intent_path.with_suffix('.tmp');temp.write_text(json.dumps({'enabled':enabled}));temp.replace(self.intent_path)

    def _spawn(self,folder):
        self.requested_project=self.current_project
        mode,config=self.config()
        extra=['--mode',mode,'--max-gb',str(config.get('max_gb',12 if mode=='line' else 48))]
        if mode=='line':extra+=['--recordings',str(config.get('lookahead',2))]
        if self.current_project:extra+=['--current-project',self.current_project]
        with (folder/'preprocess.log').open('ab') as log:
            self.process=subprocess.Popen(
                [sys.executable,'-u','-m','scripts.aoi_preprocess','--root',str(self.root),'--cache',str(self.database()),*extra],
                cwd=self.root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

    def change(self, action: str) -> dict:
        if action not in ('pause','resume'):raise ValueError('无效的预处理操作')
        with self.lock:
            folder=self.database().parent
            # Turn off automatic window advancement before reading status.
            if self.config()[0]=='line' and action=='pause':self._intent(False)
            status=self.status()['status']
            if action=='pause':
                if status=='starting':raise ValueError('正在启动，请稍后暂停')
                if status in ('running','pausing'):(folder/'STOP').touch()
            else:
                mode,_=self.config()
                if mode=='line':self._intent(True)
                if status not in (('running','pausing','starting') if mode=='line' else ('running','pausing','starting','complete')):
                    self._spawn(folder)
            return self.status()
