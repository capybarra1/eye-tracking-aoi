"""Lossless cache for the current straight-line relocalizer; never AOI geometry."""
from __future__ import annotations
import hashlib,json,shutil,sqlite3,struct,threading,time,zlib
from pathlib import Path
import cv2
import numpy as np
from scripts.aoi_feature_cache import FeatureReader,encode,decode

VERSION=hashlib.sha256(('line-scene-v1|gray480|600|.015|'+cv2.__version__+'|'+cv2.getBuildInformation()).encode()).hexdigest()
CONFIG=Path(__file__).resolve().parents[1]/'models/line_cache.json'
_lock=threading.RLock();_reader=None;_override=False;_checked=0.;_configured=None
_hits=0;_misses=0

def scene_gray(frame):
    return cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)

def frame_key(gray):
    return hashlib.sha256(VERSION.encode()+str((gray.shape,gray.dtype.str)).encode()+gray.tobytes()).hexdigest()

def configure_cache(path):
    global _reader,_override,_configured,_hits,_misses
    with _lock:
        if _reader:_reader.close()
        _reader=None;_override=True;_configured=str(path) if path else None;_hits=0;_misses=0
        if path:
            try:_reader=FeatureReader(Path(path).resolve())
            except (OSError,sqlite3.Error):pass

def _get_reader():
    global _checked,_reader,_configured
    if not _override and time.monotonic()-_checked>3:
        _checked=time.monotonic()
        try:
            cfg=json.loads(CONFIG.read_text());path=Path(cfg['database']).expanduser().resolve() if cfg.get('enabled') else None
            requested=str(path) if path else None
            if requested!=_configured or (_reader is None and path is not None):
                if _reader:_reader.close()
                _reader=None;_configured=requested
                if path and path.is_file():_reader=FeatureReader(path)
        except (OSError,ValueError,KeyError,sqlite3.Error):
            if _reader:_reader.close()
            _reader=None;_configured=None
    return _reader

def runtime_status():
    with _lock:
        reader=_get_reader()
        return dict(hits=_hits,misses=_misses,reader_ready=reader is not None)

class CachedSceneSift:
    def __init__(self):
        self.detector=cv2.SIFT_create(nfeatures=600,contrastThreshold=.015);self.hits=0
    def detectAndCompute(self,gray,mask):
        global _hits,_misses
        try:
            with _lock:
                reader=_get_reader();result=reader.get(frame_key(gray)) if reader else None
            if result is not None:
                keys,desc=result
                if mask is not None:
                    if mask.dtype!=np.uint8 or mask.shape!=gray.shape:raise ValueError('invalid mask')
                    xy=np.asarray([k.pt for k in keys],np.float32).reshape(-1,2)
                    coords=(xy+np.float32(.5)).astype(np.int32)
                    selected=np.flatnonzero(mask[coords[:,1],coords[:,0]]!=0)
                    keys=tuple(keys[i] for i in selected);desc=desc[selected].copy() if len(selected) else None
                elif desc is not None:desc=desc.copy()
                with _lock:_hits+=1
                self.hits+=1;return keys,desc
        except (OSError,sqlite3.Error,ValueError,IndexError,struct.error,zlib.error,cv2.error):pass
        with _lock:_misses+=1
        return self.detector.detectAndCompute(gray,mask)

def scene_sift():return CachedSceneSift()

class LineWriter:
    def __init__(self,path,max_bytes=12*1024**3,min_free_bytes=20*1024**3):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        self.max_bytes=max_bytes;self.min_free_bytes=min_free_bytes
        self.db=sqlite3.connect(self.path,timeout=5)
        self.db.execute('PRAGMA auto_vacuum=INCREMENTAL')
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS features(key TEXT PRIMARY KEY,data BLOB,digest TEXT,owner TEXT)')
        self.db.execute('CREATE INDEX IF NOT EXISTS feature_owner ON features(owner)');self.db.commit()
        self.detector=cv2.SIFT_create(nfeatures=600,contrastThreshold=.015)
    def retire_owners(self,owners):
        # Only explicit completed/skipped queue members. Never inferred from age.
        self.db.executemany('DELETE FROM features WHERE owner=?',[(str(owner),) for owner in owners]);self.db.commit()
        self.db.execute('PRAGMA incremental_vacuum(256)');self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    def prepare(self,gray,owner=''):
        key=frame_key(gray)
        if self.db.execute('SELECT 1 FROM features WHERE key=?',(key,)).fetchone():return False
        keys,desc=self.detector.detectAndCompute(gray,None);data=encode(keys,desc)
        pages=self.db.execute('PRAGMA page_count').fetchone()[0]-self.db.execute('PRAGMA freelist_count').fetchone()[0]
        page_size=self.db.execute('PRAGMA page_size').fetchone()[0]
        if pages*page_size+len(data)+16384>self.max_bytes:
            raise OSError('直线缓存预算已用满；已有缓存可用，标注可继续。完成录像后继续预处理可回收其缓存。')
        if shutil.disk_usage(self.path.parent).free<len(data)+self.min_free_bytes:
            raise OSError('硬盘剩余空间不足，已暂停预处理以保留安全余量；标注和视频未修改。')
        self.db.execute('INSERT OR IGNORE INTO features VALUES (?,?,?,?)',(key,data,hashlib.sha256(data).hexdigest(),str(owner)));self.db.commit();return True
    def close(self):self.db.close()
