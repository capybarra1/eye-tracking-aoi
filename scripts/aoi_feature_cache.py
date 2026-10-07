"""Lossless, content-addressed SIFT preprocessing; never stores AOI decisions.

Only the structure tracker's exact (1400, .025) profile is cached. OpenCV SIFT
retains keypoints before applying its pixel mask, so a precomputed full-frame
result can be filtered with the current AOI mask without reusing old AOI shapes.
"""
from __future__ import annotations
import hashlib,json,shutil,sqlite3,struct,threading,time,zlib
from pathlib import Path
import cv2
import numpy as np

VERSION=hashlib.sha256(('screen-sift-v1|1400|.025|'+cv2.__version__+'|'+cv2.getBuildInformation()).encode()).hexdigest()
CONFIG=Path(__file__).resolve().parents[1]/'models/feature_cache.json'
_lock=threading.RLock();_reader=None;_override=False;_checked=0.;_configured=None
_status_checked=0.;_status={}

def cache_status():
 global _status_checked,_status
 with _lock:
  if time.monotonic()-_status_checked<3:return dict(_status)
  _status_checked=time.monotonic();reader=_get_reader()
  _status={'enabled':reader is not None}
  if reader is not None:
   try:
    progress=json.loads((reader.path.parent/'progress.json').read_text())
    _status.update({k:progress.get(k) for k in ('status','subject','segment','frames_this_run','recordings','error')})
   except (OSError,ValueError):_status['status']='ready'
  return dict(_status)

def frame_key(gray):
 return hashlib.sha256(VERSION.encode()+str((gray.shape,gray.dtype.str)).encode()+gray.tobytes()).hexdigest()

def encode(keys,desc):
 attrs=np.array([[*k.pt,k.size,k.angle,k.response,k.octave,k.class_id] for k in keys],dtype='<f8').reshape(-1,7)
 if desc is None:desc=np.empty((0,128),np.float32)
 compact=np.array_equal(desc,desc.astype(np.uint8).astype(np.float32))
 raw=struct.pack('<IB',len(keys),int(compact))+attrs.tobytes()+desc.astype('u1' if compact else '<f4').tobytes()
 return zlib.compress(raw,1)

def decode(data):
 obj=zlib.decompressobj();raw=obj.decompress(data,8_000_000)
 if not obj.eof or obj.unused_data:raise ValueError('invalid cache compression')
 count,compact=struct.unpack('<IB',raw[:5])
 if count>10000 or compact not in (0,1):raise ValueError('invalid feature count')
 dtype='u1' if compact else '<f4';offset=5+count*56
 if len(raw)!=offset+count*128*np.dtype(dtype).itemsize:raise ValueError('invalid feature bytes')
 attrs=np.frombuffer(raw,dtype='<f8',count=count*7,offset=5).reshape(-1,7)
 if not np.isfinite(attrs).all():raise ValueError('invalid keypoints')
 keys=tuple(cv2.KeyPoint(float(x),float(y),float(size),float(angle),float(response),int(octave),int(cid)) for x,y,size,angle,response,octave,cid in attrs)
 desc=np.frombuffer(raw,dtype=dtype,count=count*128,offset=offset).reshape(-1,128).astype(np.float32)
 return keys,desc if count else None

class FeatureReader:
 def __init__(self,path):
  self.path=Path(path);self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,check_same_thread=False,timeout=.01)
  self.db.execute('PRAGMA query_only=ON');self.cached_key=None;self.cached=None
 def get(self,key):
  if key==self.cached_key:return self.cached
  row=self.db.execute('SELECT data,digest FROM features WHERE key=?',(key,)).fetchone()
  if row is None:return None
  if hashlib.sha256(row[0]).hexdigest()!=row[1]:raise ValueError('cache checksum mismatch')
  result=decode(row[0]);self.cached_key=key;self.cached=result;return result
 def close(self):self.db.close()

def configure_cache(path):
 global _reader,_override,_configured
 with _lock:
  if _reader:_reader.close()
  _reader=None;_override=True;_configured=str(path) if path else None
  if path:
   try:_reader=FeatureReader(Path(path).resolve())
   except (OSError,sqlite3.Error):pass

def _get_reader():
 global _checked,_reader,_configured
 if not _override and time.monotonic()-_checked>5:
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

class CachedScreenSift:
 def __init__(self):
  self.detector=cv2.SIFT_create(nfeatures=1400,contrastThreshold=.025);self.hits=0
 def detectAndCompute(self,gray,mask):
  try:
   with _lock:
    reader=_get_reader();result=reader.get(frame_key(gray)) if reader is not None else None
   if result is not None:
    keys,desc=result
    if mask is not None:
     if mask.dtype!=np.uint8 or mask.shape!=gray.shape:raise ValueError('invalid mask')
     # Same float32 addition and truncation as KeyPointsFilter::runByPixelsMask.
     xy=np.asarray([k.pt for k in keys],np.float32).reshape(-1,2)
     coords=(xy+np.float32(.5)).astype(np.int32)
     selected=np.flatnonzero(mask[coords[:,1],coords[:,0]]!=0)
     keys=tuple(keys[i] for i in selected);desc=desc[selected].copy() if len(selected) else None
    elif desc is not None:desc=desc.copy()
    self.hits+=1;return keys,desc
  except (OSError,sqlite3.Error,ValueError,IndexError,struct.error,zlib.error,cv2.error):pass
  return self.detector.detectAndCompute(gray,mask)

def screen_sift():return CachedScreenSift()

class FeatureWriter:
 def __init__(self,path,max_bytes=48*1024**3,min_free_bytes=20*1024**3):
  self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
  self.max_bytes=max_bytes;self.min_free_bytes=min_free_bytes
  self.db=sqlite3.connect(self.path,timeout=5)
  self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=NORMAL')
  self.db.execute('CREATE TABLE IF NOT EXISTS features(key TEXT PRIMARY KEY,data BLOB,digest TEXT)');self.db.commit()
  self.detector=cv2.SIFT_create(nfeatures=1400,contrastThreshold=.025)
 def prepare(self,gray):
  key=frame_key(gray)
  if self.db.execute('SELECT 1 FROM features WHERE key=?',(key,)).fetchone():return False
  keys,desc=self.detector.detectAndCompute(gray,None);data=encode(keys,desc)
  size=self.path.stat().st_size;wal=Path(str(self.path)+'-wal')
  if wal.exists():size+=wal.stat().st_size
  if size+len(data)+8192>self.max_bytes or shutil.disk_usage(self.path.parent).free<len(data)+self.min_free_bytes:
   raise OSError('缓存空间达到上限；已保留完成部分，未修改视频或标注')
  self.db.execute('INSERT OR IGNORE INTO features VALUES (?,?,?)',(key,data,hashlib.sha256(data).hexdigest()));self.db.commit();return True
 def close(self):self.db.close()
