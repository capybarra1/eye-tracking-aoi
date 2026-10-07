"""Loopback-only HTTP interface for the supervised tracker."""
from __future__ import annotations

from collections import OrderedDict
import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, quote

from scripts.aoi_image_adjustments import display_frame, apply_settings
import cv2

from scripts.fast_aoi_tracker import load_prompt_polygons, expand_polygon_outward
from scripts.supervised_aoi import Session, VideoSource, Tracker
from scripts.aoi_runtime_config import configure_detector
from scripts.aoi_scrub_preview import preview_frame
from scripts.aoi_projects import Projects, TIMING
from scripts.aoi_batch import BatchWorker
from scripts.aoi_progress import subject_directory
from scripts.aoi_slice_timing import timing_preview,adjust_frames
from scripts.aoi_workflow_api import ReviewWorker, all_issues
from scripts.aoi_async_saves import AnnotationSaves
from scripts.aoi_preprocess_control import PreprocessControl


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,session,projects=None,detector_config=None):
        super().__init__(address,Handler)
        self.session=session;self.lock=threading.RLock();self.revision=0
        self.projects=projects;self.save_error=None
        self.saves=AnnotationSaves(projects) if projects else None
        self.preview_stop=None;self.preview_image=None;self.scrub_images=OrderedDict()
        self.preprocessing=PreprocessControl(Path(__file__).resolve().parents[1])
        config=Path(detector_config) if detector_config is not None else Path(__file__).resolve().parents[1]/'models/aoi_rescue.json'
        Tracker.default_detector,self.smart_recovery=configure_detector(config)
        Tracker.default_detector_parallel=bool(self.smart_recovery.get('enabled') and self.smart_recovery.get('device')=='mps')
        self.smart_recovery['cpu_gpu_parallel']=Tracker.default_detector_parallel
        if Tracker.default_detector_parallel:
            self.smart_recovery['message']='CPU + GPU 并行 · 平板自动找回（试运行）。'
        self.batch=BatchWorker(self);self.review=ReviewWorker(self)

    def server_close(self):
        self.batch.request_pause();self.review.cancel()
        if self.batch.thread:self.batch.thread.join(timeout=15)
        if self.saves:self.saves.close()
        super().server_close()

    def save_annotations(self, session, reason):
        return self.saves.request(session,reason) if self.saves else None


def step_preview(session, token: str, frames: int = 1, stop_requested=None):
    """Compute every frame; batch only preview delivery, stopping at every guard."""
    if type(frames) is not int or not 1 <= frames <= 10:
        raise ValueError('预览步长必须为 1 至 10 帧')
    for _ in range(frames):
        if stop_requested and stop_requested():
            session.pause();break
        session.step(token)
        if not session.running:
            break
        if stop_requested and stop_requested():
            session.pause();break


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass

    def send(self,status,data,kind='application/json; charset=utf-8',filename=None):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type',kind)
        if filename:self.send_header('Content-Disposition',"attachment; filename*=UTF-8''"+quote(filename,safe=''))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.send_header('Content-Length',str(len(data)));self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass

    def allowed(self):
        host=self.headers.get('Host','')
        allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        origin=self.headers.get('Origin')
        return host in allowed and (origin is None or origin in {f'http://{h}' for h in allowed})

    def payload(self):
        s=self.server.session
        state=s.snapshot();state['revision']=self.server.revision
        state['smart_recovery']=self.server.smart_recovery
        self.server.preprocessing.current_project=self.server.projects.current if self.server.projects else None
        state['preprocessing']=self.server.preprocessing.status()
        state['batch']=self.server.batch.snapshot();state['review']=self.server.review.snapshot()
        self.server.review.overlay(s,state)
        if self.server.projects:
            p=self.server.projects.index[self.server.projects.current]
            state['project']={k:p[k] for k in ('id','subject','video','kind')}
            state['project']['label']=p.get('label','')
            state['project']['import_notes']=p.get('import_notes',[])
            state['saved']=self.server.saves.status(s)
            if self.server.save_error:state['saved']={**state['saved'],'status':'error','error':self.server.save_error}
        cached=self.server.preview_image
        if cached is None or cached[0] is not s.source or cached[1]!=s.cursor:
            ok,data=cv2.imencode('.jpg',display_frame(s.source,s.cursor),[cv2.IMWRITE_JPEG_QUALITY,80])
            if not ok:raise ValueError('预览图生成失败')
            cached=(s.source,s.cursor,'data:image/jpeg;base64,'+base64.b64encode(data).decode())
            self.server.preview_image=cached
        state['image']=cached[2]
        return state

    def do_GET(self):
        if not self.allowed():return self.send(403,{'error':'仅允许本地同源访问'})
        path=urlsplit(self.path).path
        if path=='/api/preprocess-status':
            self.server.preprocessing.current_project=self.server.projects.current if self.server.projects else None
            return self.send(200,self.server.preprocessing.status())
        if path=='/':return self.send(200,Path(__file__).with_suffix('.html').read_bytes(),'text/html; charset=utf-8')
        if path=='/aoi_image_adjustments.js':return self.send(200,Path(__file__).with_name('aoi_image_adjustments.js').read_bytes(),'text/javascript; charset=utf-8')
        if path=='/aoi_slice_timing.js':return self.send(200,Path(__file__).with_name('aoi_slice_timing.js').read_bytes(),'text/javascript; charset=utf-8')
        if path=='/aoi_scrub.js':return self.send(200,Path(__file__).with_name('aoi_scrub.js').read_bytes(),'text/javascript; charset=utf-8')
        if path=='/aoi_workflow.js':return self.send(200,Path(__file__).with_name('aoi_workflow.js').read_bytes(),'text/javascript; charset=utf-8')
        if path=='/aoi_drag.js':return self.send(200,Path(__file__).with_name('aoi_drag.js').read_bytes(),'text/javascript; charset=utf-8')
        if path=='/aoi_directory.js':return self.send(200,Path(__file__).with_name('aoi_directory.js').read_bytes(),'text/javascript; charset=utf-8')
        if path=='/aoi_projects.js':return self.send(200,Path(__file__).with_name('aoi_projects.js').read_bytes(),'text/javascript; charset=utf-8')
        manager=self.server.projects
        if manager and path=='/api/save-status':
            with self.server.lock:
                expected=parse_qs(urlsplit(self.path).query).get('project',[manager.current])[0]
                if expected!=manager.current:return self.send(409,{'error':'已切换被试'})
                return self.send(200,{'project_id':manager.current,'saved':self.server.saves.status(self.server.session)})
        if manager and path=='/api/issues':
            with self.server.lock:return self.send(200,all_issues(manager))
        if manager and path=='/api/directory':return self.send(200,subject_directory(manager))
        if manager and path=='/api/catalog':return self.send(200,manager.catalog())
        if manager and path=='/api/analysis':return self.send(200,manager.analysis_state())
        if manager and path=='/api/analysis/download':
            result=manager.analysis_state()
            if result['status']=='complete':return self.send(200,Path(result['csv']).read_bytes(),'text/csv; charset=utf-8')
        if manager and path.startswith('/api/timing/'):
            try:
                subject,index=map(int,path.removeprefix('/api/timing/').split('/'))
                slide=manager.eligible[subject]['source_slide']
                file=sorted(TIMING.glob(f'slide{slide:02d}_image*.png'))[index]
                return self.send(200,file.read_bytes(),'image/png')
            except (ValueError,KeyError,IndexError):return self.send(404,{'error':'无切片时间截图'})
        if manager and path in ('/api/review/download','/api/annotations/archive','/api/annotations/table'):
            with self.server.lock:
                expected=parse_qs(urlsplit(self.path).query).get('project',[manager.current])[0]
                if expected!=manager.current:return self.send(409,{'error':'已切换被试，请刷新后下载'})
                saved=self.server.saves.status(self.server.session)
                if saved.get('status')=='saving':return self.send(409,{'error':'存档正在更新，请稍后下载'})
                key='csv_name' if path.endswith('/table') else 'archive_name'
                if saved.get('status')=='saved':
                    file=self.server.session.folder/'exports'/saved[key]
                    kind='text/csv; charset=utf-8' if key=='csv_name' else 'application/json; charset=utf-8'
                    return self.send(200,file.read_bytes(),kind,filename=file.name)
        if path=='/api/timing-preview':
            with self.server.lock:
                try:
                    query=parse_qs(urlsplit(self.path).query);current=manager.current if manager else ''
                    if query.get('project',[''])[0]!=current or query.get('segment',[''])[0]!=self.server.session.segment or int(query.get('revision',['-1'])[0])!=self.server.revision:
                        return self.send(409,{'error':'项目或标注已变化，请重新打开时间调整'})
                    if self.server.session.running or self.server.batch.active:raise ValueError('请先暂停标注')
                    return self.send(200,timing_preview(self.server.session,int(query['index'][0])))
                except (ValueError,KeyError,IndexError) as e:return self.send(400,{'error':str(e)})
                except Exception as e:return self.send(500,{'error':'预览暂不可用：'+str(e)})
        if path=='/api/preview':
            with self.server.lock:
                try:
                    query=parse_qs(urlsplit(self.path).query)
                    current=manager.current if manager else ''
                    if query.get('project',[''])[0]!=current or query.get('segment',[''])[0]!=self.server.session.segment or int(query.get('revision',['-1'])[0])!=self.server.revision:
                        return self.send(409,{'error':'项目或标注已变化，请松手重新定位'})
                    return self.send(200,preview_frame(self.server,int(query['index'][0])))
                except (ValueError,KeyError,IndexError) as e:return self.send(400,{'error':str(e)})
                except Exception as e:return self.send(500,{'error':'预览暂不可用：'+str(e)})
        if path=='/api/state':
            with self.server.lock:
                try:return self.send(200,self.payload())
                except Exception as e:return self.send(500,{'error':str(e)})
        self.send(404,{'error':'未找到'})

    def save_annotations(self, session, reason):
        if not self.server.projects:return None
        try:
            result=self.server.save_annotations(session,reason)
            self.server.save_error=None
            return result
        except Exception as e:
            self.server.save_error='标注存档和数据表保存失败，请重试「保存标注」：'+str(e)
            raise RuntimeError(self.server.save_error) from e

    def do_POST(self):
        if not self.allowed() or self.headers.get('X-AOI-Client')!='1':
            return self.send(403,{'error':'仅允许本地审核界面提交操作'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=64_000_000:raise ValueError('请求大小不合法（最多64MB）')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):raise ValueError('无效请求')
        except (ValueError,TypeError) as e:return self.send(400,{'error':str(e)})
        action=urlsplit(self.path).path.removeprefix('/api/')
        if action=='preprocess-control':
            self.server.preprocessing.current_project=self.server.projects.current if self.server.projects else None
            try:return self.send(200,self.server.preprocessing.change(data.get('action')))
            except (OSError,ValueError,KeyError) as e:return self.send(400,{'error':str(e)})
        if action=='pause-request':
            # No database mutation here: let the active frame finish, then stop
            # its preview loop. A replaced Session cannot inherit this request.
            s=self.server.session
            current=self.server.projects.current if self.server.projects else None
            if data.get('project_id')!=current:return self.send(409,{'error':'已切换被试'})
            if data.get('token') and data['token']!=s.token:return self.send(409,{'error':'运行已变化'})
            self.server.preview_stop=s
            self.server.batch.request_pause()
            return self.send(200,{'ok':True})
        if action=='pause':self.server.batch.request_pause()
        wait_for_save=None
        with self.server.lock:
            s=self.server.session
            try:
                if self.server.batch.active and action not in ('pause','batch_skip'):return self.send(409,{'error':'批量运行中，请先暂停'})
                if self.server.review.active and action not in ('pause','review_cancel'):raise ValueError('正在生成复核预览，请等待或取消')
                if self.server.review.draft and action in ('partition','correct','reset','continue','start','step','approve','batch_start','project_open','project_create','review_import','select','timing_adjust','image_adjustments'):
                    raise ValueError('请先退出问题复核，再执行此操作')
                if action not in ('pause','batch_skip') and data.get('revision')!=self.server.revision:
                    raise ValueError('画面已更新，请重试当前操作；不要同时使用两个审核页面')
                result=None;notice=None;save_reason=None;save_attempted=False;was_running=s.running
                if action=='batch_start':
                    s.pause();self.server.batch.start()
                elif action=='batch_skip':
                    self.server.batch.skip_current(data.get('project_id'),data.get('position'));notice='已保存并跳过当前录像；不会把未处理部分算作完成。'
                elif action=='batch_restore':self.server.batch.restore(data.get('project_id'));notice='已放回待处理队列，可继续跑全部。'
                elif action=='review_preview':self.server.review.start(data)
                elif action=='review_commit':self.server.review.commit();save_reason='局部复核保存'
                elif action=='review_cancel':self.server.review.cancel()
                elif action=='start':self.server.preview_stop=None;s.start()
                elif action=='continue':self.server.preview_stop=None;s.continue_simple(data.get('polygons'),data.get('visible'))
                elif action=='step':step_preview(s,data['token'],data.get('frames',1),lambda:self.server.preview_stop is s)
                elif action=='pause':s.pause()
                elif action=='image_adjustments':
                    current=self.server.projects.current if self.server.projects else ''
                    if data.get('project_id')!=current:raise ValueError('被试已切换，请重试')
                    apply_settings(s,data.get('settings'))
                    self.server.preview_image=None;self.server.scrub_images.clear()
                elif action=='timing_adjust':
                    current=self.server.projects.current if self.server.projects else ''
                    if data.get('project_id')!=current or data.get('segment')!=s.segment:raise ValueError('被试或切片已切换，请重新打开时间调整')
                    if s.running:raise ValueError('请先暂停标注')
                    adjust_frames(s,data.get('first'),data.get('last'));save_reason='调整切片范围'
                    notice='切片范围已保存；已有标注保留，新增部分留待标注。'
                    batch=self.server.batch
                    for item in batch.state.get('items',[]):
                        if item.get('project_id')==current and item.get('status')=='complete':item['status']='pending'
                    batch.persist()
                elif action=='select':s.select(str(data['segment']))
                elif action=='seek':s.seek(int(data['index']))
                elif action=='partition':s.set_partition(data.get('enabled'));save_reason='切换分区方式'
                elif action=='correct':s.correct(data['polygons'],data['visible'])
                elif action=='reset':s.reset_from_here(data['polygons'],data['visible'])
                elif action=='approve':s.approve()
                elif action in ('save','export'):
                    if self.server.review.draft and data.get('polygons') is not None:raise ValueError('复核修改请使用保存此段')
                    s.pause()
                    if data.get('polygons') is not None:s.correct(data['polygons'],data['visible'])
                    save_reason='手动保存'
                elif action in ('project_open','project_create','review_import'):
                    manager=self.server.projects
                    if manager is None:raise ValueError('当前服务未启用项目管理')
                    s.pause();save_attempted=True;self.save_annotations(s,'切换项目前保存')
                    if action=='project_open':new=manager.open(data['project_id'])
                    elif action=='project_create':new=manager.create(data)
                    else:new=manager.import_review(data['package'],data.get('video',''))
                    self.server.session=new;s.close();s.source.close();s=new
                    self.server.save_error=None
                    if action=='review_import':save_reason='导入标注存档'
                    notice='已打开独立复核副本；视频指纹、时间戳及坐标校验通过。' if action=='review_import' else '本地项目已打开；新切片需要调整起始AOI。'
                elif action=='analyze':
                    if self.server.projects is None:raise ValueError('未启用项目管理')
                    self.server.projects.analyze(s);notice='指标计算已启动，结果基于已保存的AOI快照。'
                else:raise ValueError('未知操作')
                if action in ('pause','correct','reset','approve','select','analyze'):
                    save_reason={'pause':'暂停保存','correct':'手工修正','reset':'从当前位置重标','approve':'人工确认','select':'切换切片','seek':'回看暂停','analyze':'计算指标前保存'}[action]
                elif action=='seek' and was_running:save_reason='回看暂停'
                elif action=='continue' and data.get('polygons') is not None:save_reason='保存起始轮廓'
                elif action=='step' and not s.running:
                    save_reason='切片结束' if s.cursor==s.last_index else '异常暂停' if s.mode=='issue' else '检查点暂停'
                if save_reason:
                    save_attempted=True;result=self.save_annotations(s,save_reason)
                self.server.revision+=1
                if action in ('save','export','review_commit') and result:
                    wait_for_save=(s,result['generation'])
                payload=self.payload()
                if result:payload['saved']=result
                if notice:payload['notice']=notice
                if wait_for_save is None:return self.send(200,payload)
            except Exception as e:
                s.pause()
                if action=='step' and not isinstance(e,(ValueError,KeyError)):
                    s.mode='issue';s.reason='计算错误，已暂停：'+str(e);s.tracker=None;s._save()
                if action=='step' and not locals().get('save_attempted',False):
                    try:self.save_annotations(s,'计算异常暂停')
                    except Exception:pass
                self.server.revision+=1
                try:payload=self.payload()
                except Exception:payload={}
                return self.send(400,{'error':str(e),'state':payload})
        # Explicit Save waits for the pair, but never occupies the tracking/UI
        # lock while the background worker serializes and writes the files.
        if wait_for_save is not None:
            try:
                self.server.saves.wait(*wait_for_save)
                with self.server.lock:return self.send(200,self.payload())
            except Exception as e:
                with self.server.lock:return self.send(400,{'error':str(e),'state':self.payload()})


def main():
    parser=argparse.ArgumentParser(description='AOI 切片审核工具：启动后保持暂停')
    parser.add_argument('--manifest',type=Path,default=Path('outputs/aoi_workflow_20260928/timing_v3/demo_segments.json'))
    parser.add_argument('--timestamps',type=Path,default=Path('outputs/aoi_demo_video/source_pts.json'))
    parser.add_argument('--template',type=Path,default=Path('outputs/aoi_audit_20260928/screen_structure_v2/user_prompts.json'))
    parser.add_argument('--workspace',type=Path,default=Path('outputs/aoi_demo_session'))
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--projects-root',type=Path,default=None)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    manifest=json.loads(args.manifest.read_text())
    if not manifest:parser.error('清单为空')
    source=VideoSource(Path(manifest[0]['video_path']),args.timestamps)
    shapes=load_prompt_polygons(args.template,source.size)
    # This is a reference template, never automatically approved on another frame.
    shapes['tablet']=[expand_polygon_outward(p,36.) for p in shapes['tablet']]
    template={key:[p.tolist() for p in values] for key,values in shapes.items()}
    session=Session(manifest,args.workspace,source,template)
    server=Server(('127.0.0.1',args.port),session,Projects(session,args.timestamps,args.projects_root))
    print(f'本地审核：http://127.0.0.1:{server.server_port}  数据：{args.workspace.resolve()}  初始状态：暂停',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();server.session.close();server.session.source.close()


if __name__=='__main__':main()
