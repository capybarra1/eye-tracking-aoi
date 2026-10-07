"""Resumable local batch worker. Never overwrites an existing annotation frame."""
from __future__ import annotations
import copy
import json
import sqlite3
import threading
import time
from pathlib import Path
from scripts.supervised_aoi import validate_polygons, four_points
from scripts.aoi_horizontal import partition_polygons, center_height, center_slope
from scripts.aoi_auto_retry import tracking_failed, prepare_retry, run_retry, commit_retry, clone_source

UNTRUSTED='起始轮廓待确认'


def build_queue(manager, excluded_subjects=()):
    catalog=manager.catalog();items=[];seen=set()
    old_projects={v.get('project_id') for v in catalog['videos'] if v.get('annotation_history',{}).get('old_aoi_present')}
    for p in sorted(catalog['projects'],key=lambda p:(p['id']!=manager.current,p['subject'],p['id'])):
        if p['subject'] not in manager.eligible or p['subject'] in excluded_subjects or p['kind']!='标注':continue
        if p['id'] in old_projects and p['id']!=manager.current:continue
        items.append(dict(project_id=p['id'],subject=p['subject'],label=p.get('label') or f"{p['subject']}号",status='pending'));seen.add(p['id'])
    for entry in catalog['videos']:
        if entry['subject'] not in manager.eligible or entry['subject'] in excluded_subjects or not entry.get('manifest'):continue
        if entry.get('annotation_history',{}).get('old_aoi_present'):continue
        if entry.get('project_id') in seen:continue
        items.append(dict(project_id=entry.get('project_id'),subject=entry['subject'],label=entry.get('name',''),entry=entry,status='pending'))
        if entry.get('project_id'):seen.add(entry['project_id'])
    return items


def next_gap(s):
    # Called at slice transitions / replay boundaries, not once per frame.
    for segment,row in s.slices.items():
        expected=row['first']
        for frame, in s.db.execute('SELECT frame FROM records WHERE segment=? AND frame BETWEEN ? AND ? ORDER BY frame',(segment,row['first'],row['last'])):
            if frame>expected:return segment,expected
            expected=frame+1
        if expected<=row['last']:return segment,expected
    return None


def fill_next(s):
    """Add one missing frame; reuse all existing records and manual anchors."""
    direct=s.running and s.cursor<s.last_index and not s.get_record(s.cursor+1)
    if not direct:
        gap=next_gap(s)
        if gap is None:s.pause();return False
        segment,index=gap
        if s.segment!=segment:s.select(segment)
        if index==s.first_index:
            s.seek(index);snapshot=s.snapshot();key=s.get_anchor(index)
            polygons=key['polygons'] if key else snapshot['polygons'];visible=key['visible'] if key else snapshot['visible']
            if not key and s._get('horizontal_mode') is True and not polygons.get('partition'):
                boundary={'screen':[four_points(polygons['screen'],s.source.size)]}
                polygons=partition_polygons(s.source.size,center_height(boundary,s.source.size),s._get('partition_gap') if s._get('partition_gap') is not None else None,slope=center_slope(boundary,s.source.size))
            polygons=validate_polygons(polygons,visible,s.source.size)
            s._put(index,polygons,visible,[] if key else [UNTRUSTED]);s._save()
            return True
        s.seek(index-1);s.mode='paused';s.start(require_review=False)
    s.step(s.token,defer_issues=True)
    return True


class BatchWorker:
    def __init__(self,server):
        self.server=server;self.stop=threading.Event();self.thread=None
        self.path=server.projects.root/'batch_state.json' if server.projects else None
        self.state=json.loads(self.path.read_text()) if self.path and self.path.exists() else dict(status='idle',items=[],position=0)
        if self.state.get('status') in ('running','preparing'):self.state['status']='paused'
        self.skip_path=server.projects.root/'manual_skips.json' if server.projects else None
        self.skips=json.loads(self.skip_path.read_text()) if self.skip_path and self.skip_path.exists() else {}
        scope_path=server.projects.root/'batch_scope.json' if server.projects else None
        self.scope=json.loads(scope_path.read_text()) if scope_path and scope_path.exists() else {}
        excluded=self.scope.get('excluded_subjects',[])
        if not isinstance(excluded,list) or any(type(s) is not int for s in excluded):
            raise ValueError('批量范围配置无效')
        self.excluded_subjects=set(excluded)
        self.apply_scope()
        self.active=False
    def apply_scope(self):
        if not self.excluded_subjects:return
        items=self.state['items'];position=self.state['position']
        self.state['position']=sum(i['subject'] not in self.excluded_subjects for i in items[:position])
        self.state['items']=[i for i in items if i['subject'] not in self.excluded_subjects]
    def persist(self):
        if self.path:
            temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(self.state,ensure_ascii=False,indent=2));temp.replace(self.path)
    def snapshot(self):
        return {**self.state,'active':self.active,'done':sum(i['status']=='complete' for i in self.state['items']),
                'blocked':sum(i['status']=='blocked' for i in self.state['items']),'total':len(self.state['items']),
                'skipped':sum(i['status']=='skipped' for i in self.state['items']),'manual_skips':list(self.skips.values()),
                'scope_label':self.scope.get('label',''),'excluded_subjects':sorted(self.excluded_subjects)}
    def start(self):
        if self.active:return
        if not self.server.projects:raise ValueError('未启用项目管理')
        from scripts.aoi_range_review import signature
        s=self.server.session;failed=s._get('auto_retry_failed')
        if failed and failed.get('segment')==s.segment and signature(s,failed['start'],failed['end'])==failed['signature']:
            raise ValueError('这段已经自动重试2次；请先调整当前直线，再继续自动标注')
        if self.state['status'] not in ('paused','error') or not self.state['items']:
            self.state=dict(status='running',items=build_queue(self.server.projects,self.excluded_subjects),position=0)
            for item in self.state['items']:
                if item.get('project_id') in self.skips:item['status']='skipped'
        else:
            # Refresh pending imports when the user has added a new device batch.
            def identity(item):return item.get('project_id') or item.get('entry',{}).get('key')
            fresh=build_queue(self.server.projects,self.excluded_subjects)
            valid={identity(i) for i in fresh}
            completed=self.state['items'][:self.state['position']]
            pending=[i for i in self.state['items'][self.state['position']:] if identity(i) in valid]
            known={identity(i) for i in completed+pending}
            pending += [i for i in fresh if identity(i) not in known]
            current=self.server.projects.current
            pending.sort(key=lambda i:(i.get('project_id')!=current,i.get('entry',{}).get('annotation_status',{}).get('priority',1),i['subject']))
            for item in pending:
                if item.get('project_id') in self.skips:item['status']='skipped'
            self.state['items']=completed+pending;self.state['position']=len(completed)
        self.apply_scope()
        self.state['horizontal_mode']=bool(self.server.session._get('horizontal_mode'))
        self.state['partition_gap']=self.server.session._get('partition_gap')
        self.stop.clear();self.active=True;self.state['status']='running';self.state.pop('error',None);self.persist()
        self.state.pop('needs_attention',None);self.state.pop('retry',None)
        self.thread=threading.Thread(target=self.run,daemon=True,name='aoi-batch');self.thread.start()
    def save_skips(self):
        if self.skip_path:
            temp=self.skip_path.with_suffix('.tmp');temp.write_text(json.dumps(self.skips,ensure_ascii=False,indent=2));temp.replace(self.skip_path)

    def skip_current(self,project_id,position):
        # Called with server.lock. Identity and queue position replace frame
        # revision checks: a running video naturally advances between clicks.
        if type(position) is not int or position!=self.state['position'] or not 0<=position<len(self.state['items']):
            raise ValueError('队列已变化，请刷新后重试')
        item=self.state['items'][position]
        if self.state['status'] not in ('running','paused','retrying') or project_id!=self.server.projects.current or item.get('project_id')!=project_id or item['status'] in ('complete','skipped'):
            raise ValueError('当前录像已变化，未跳过任何数据')
        s=self.server.session;s.pause()
        reason='遮挡看不清（手动跳过）'
        record=dict(project_id=project_id,subject=item['subject'],label=item['label'],segment=s.segment,frame=s.cursor,reason=reason)
        s._event('manual_skip_recording',record);s._save()
        try:self.server.save_annotations(s,'手动跳过前保存')
        except Exception:
            self.request_pause();raise
        self.skips[project_id]=record;self.save_skips()
        item.update(status='skipped',skip_reason=reason)
        if not self.active:self.state['position']+=1
        self.persist()

    def restore(self,project_id):
        if self.active:raise ValueError('请先暂停全部，再恢复跳过的录像')
        if project_id not in self.skips:raise ValueError('未找到这份跳过记录')
        if self.skips[project_id]['subject'] in self.excluded_subjects:raise ValueError('该被试不在当前补标范围内')
        self.skips.pop(project_id);self.save_skips()
        item=next((i for i in self.state['items'] if i.get('project_id')==project_id),None)
        if item:
            pending=copy.deepcopy(item);pending['status']='pending';pending.pop('skip_reason',None)
            # Move this recording to the unprocessed tail, without rerunning
            # unrelated completed recordings.
            old_position=self.state['position']
            prior=sum(i.get('project_id')==project_id for i in self.state['items'][:old_position])
            self.state['items']=[i for i in self.state['items'] if i.get('project_id')!=project_id]+[pending]
            self.state['position']=old_position-prior
            self.state['status']='paused';self.persist()

    def request_pause(self):self.stop.set()

    def recover(self,item,s,start):
        server=self.server;source=None;plan=None
        def cancelled():return self.stop.is_set() or item['status']=='skipped' or server.session is not s
        def progress(attempt,frame):
            with server.lock:
                if cancelled():raise InterruptedError()
                self.state['retry']=dict(start=start,frame=frame,attempt=attempt,max_attempts=2)
                s.reason=f'自动回退重试 {attempt}/2：正在重新定位，找回后继续';server.revision+=1
        try:
            with server.lock:
                plan=prepare_retry(s,start)
                previous=s._get('auto_retry_failed')
                if previous and previous.get('segment')==s.segment and previous.get('start')==start and previous.get('signature')==plan['signature']:
                    raise ValueError('此处已重试2次；请修正红色片段起点后继续')
                s.seek(start);self.state['status']='retrying';self.state['retry']=dict(start=start,frame=start,attempt=1,max_attempts=2);self.persist();server.revision+=1
                source=clone_source(s.source)
            result=run_retry(source,plan,cancelled,progress)
            with server.lock:
                if cancelled():return False
                if not result['success']:raise ValueError(result['error'])
                commit_retry(s,plan,result);s._set('auto_retry_failed',None);s._save()
                self.state['status']='running';self.state.pop('retry',None);server.revision+=1
                server.save_annotations(s,'自动重试局部保存');self.persist();return True
        except InterruptedError:return False
        except (OSError,ValueError) as exc:
            with server.lock:
                if cancelled():return False
                s.pause();s.cursor=start;s.mode='issue';s.reason='自动重试未能找回：'+str(exc)+'。请调整当前直线后继续。'
                if plan:s._set('auto_retry_failed',dict(segment=s.segment,start=start,end=plan['end'],signature=plan['signature']))
                s._save();self.state.update(needs_attention=True,error=s.reason);self.state.pop('retry',None)
                item['status']='pending';self.stop.set();server.revision+=1;server.save_annotations(s,'自动重试失败暂停');return False
        finally:
            if source is not None and hasattr(source,'cap'):source.close()

    def run(self):
        server=self.server
        try:
            while self.state['position']<len(self.state['items']) and not self.stop.is_set():
                item=self.state['items'][self.state['position']]
                if item['status'] in ('complete','skipped'):self.state['position']+=1;continue
                # Prepare a separate session off-lock; preview/pause remain responsive
                # while a new video is hashed/indexed. UI mutations stay disabled.
                if item.get('project_id')!=server.projects.current:
                    self.state['status']='preparing';self.state['label']=item['label'];self.persist()
                    manager=copy.copy(server.projects);manager.index=copy.deepcopy(server.projects.index);manager.cancel_check=self.stop.is_set
                    try:
                        for attempt in range(2):
                            try:
                                if item.get('project_id'):new=manager.open(item['project_id'])
                                else:
                                    e=item['entry'];new=manager.create(dict(subject=e['subject'],video=e['path'],manifest=e['manifest'],split_recording=True,import_key=e.get('key')))
                                break
                            except (OSError,ValueError):
                                if attempt==1:raise
                        item['project_id']=manager.current
                    except InterruptedError:break
                    except (OSError,ValueError) as exc:
                        with server.lock:
                            item.update(status='pending',error=str(exc));self.state.update(needs_attention=True,error=f'{item["label"]}打开重试失败：{exc}')
                            self.stop.set();server.revision+=1;self.persist()
                        break
                    with server.lock:
                        server.projects.index=manager.index
                        if self.stop.is_set():new.close();new.source.close();break
                        if self.state.get('horizontal_mode'):new._set('horizontal_mode',True);new._set('partition_gap',self.state.get('partition_gap'));new._save()
                        old=server.session;server.session=new;server.projects.current=manager.current
                        old.close();old.source.close();server.revision+=1
                self.state['status']='running';item['status']='running';self.persist();last_save=time.monotonic();loss_start=None;loss_segment=server.session.segment
                with server.lock:
                    s=server.session
                    if s._get('horizontal_mode') is True and tracking_failed(s.get_record(s.cursor)):
                        loss_start=s.cursor
                        while loss_start>s.first_index and s.source.times[s.cursor]-s.source.times[loss_start]<8 and tracking_failed(s.get_record(loss_start-1)):loss_start-=1
                while not self.stop.is_set():
                    retry_start=None
                    with server.lock:
                        if item['status']=='skipped':
                            self.state['position']+=1;self.persist();break
                        s=server.session;segment=s.segment
                        server.saves.raise_if_error()
                        try:
                            for read_attempt in range(3):
                                try:more=fill_next(s);break
                                except (OSError,ValueError) as error:
                                    if read_attempt==2 or not any(word in str(error) for word in ('解码','视频','录像','读取')):raise
                                    replacement=clone_source(s.source);old_source=s.source;s.source=replacement;s.tracker=None
                                    if hasattr(old_source,'cap'):old_source.close()
                        except (OSError,ValueError) as exc:
                            # Video/geometry failures are visible queue items, never
                            # represented as a completed subject.
                            s.pause();s.mode='issue';s.reason='读取或计算失败：'+str(exc);s._save()
                            item.update(status='pending',error=str(exc));self.state.update(needs_attention=True,error=s.reason);self.stop.set();server.save_annotations(s,'批量受阻保存');more=False
                        server.revision+=1
                        if loss_segment!=s.segment:loss_start=None;loss_segment=s.segment
                        if more and s._get('horizontal_mode') is True:
                            if tracking_failed(s.get_record(s.cursor)):
                                if loss_start is None:loss_start=s.cursor
                                if s.source.times[s.cursor]-s.source.times[loss_start]>=.4 or s.cursor==s.last_index:retry_start=loss_start
                            else:loss_start=None
                        if segment!=s.segment or time.monotonic()-last_save>60 or not more:
                            server.save_annotations(s,'批量自动保存');last_save=time.monotonic()
                        if not more:
                            if self.state.get('needs_attention'):break
                            if item['status']!='blocked':item['status']='complete'
                            self.state['position']+=1;self.persist();break
                    if retry_start is not None:
                        self.recover(item,s,retry_start);loss_start=None
                    # Release the lock fairly to pause and preview requests.
                    self.stop.wait(.001)
            with server.lock:
                server.session.pause();server.save_annotations(server.session,'批量暂停保存' if self.stop.is_set() else '批量结束保存')
                self.state['status']='paused' if self.stop.is_set() else 'running';server.revision+=1
            if not self.stop.is_set():
                # All videos are computed. Finish their latest export pairs
                # outside the UI lock before reporting the batch complete.
                server.saves.wait_all()
                with server.lock:
                    self.state['status']='paused' if self.stop.is_set() else 'complete';server.revision+=1
        except Exception as exc:
            with server.lock:
                server.session.pause();server.session.mode='issue';server.session.reason='自动标注已停止：'+str(exc);server.session._save()
                self.state.update(status='error',error=str(exc),needs_attention=True);server.save_error='批量已停止：'+str(exc);server.revision+=1
        finally:
            self.active=False;self.persist()
