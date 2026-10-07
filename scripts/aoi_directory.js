'use strict';
let directoryLoading=false,directoryMarkup=null;
async function openTimelineSpan(span){
 if((typeof timingDraft!=='undefined'&&timingDraft)||busy||want||drag||workflowBusy||state.batch?.active)return;
 await go(span.start);
}
async function directoryLocation(recording,segment,index,span){
 if(projectWaiting)return;
 await projectAction(async()=>{
  if(!await save())return;await stopBatch();
  let pid=recording.project_id;
  if(!pid){
   const entry=projectCatalog?.videos.find(v=>v.key===recording.import_key);
   if(!entry)throw Error('请在添加视频中选择这份录像');
   const result=await safe('project_create',{subject:entry.subject,video:entry.path,manifest:entry.manifest,split_recording:entry.split_recording,import_key:entry.key});
   if(!result)throw Error($('feedback').textContent||'打开失败');pid=result.project.id;
  }
  if(state.project.id!==pid&&!await safe('project_open',{project_id:pid}))return;
  if(segment&&state.segment!==segment.segment&&!await safe('select',{segment:segment.segment}))return;
  $('projectDialog').close();
  if(index!=null)await go(index);
 });
}
function drawSubjectDirectory(data){
 $('subjectDirectory').replaceChildren(...data.subjects.map(subject=>{
  const section=document.createElement('section');section.className='subject-row';
  const title=document.createElement('h3');
  const named=subject.recordings.find(r=>r.label)?.label.split(' · ')[0]||`${String(subject.subject).padStart(2,'0')}号`;
  title.textContent=named;section.append(title);
  const counts=document.createElement('div');counts.className='progress-count';
  const text=document.createElement('span');text.textContent=`已知切片 ${subject.known_segments}/12 · 本工具已标 ${subject.percent}% `;
  const red=document.createElement('span');red.className='red';red.textContent=subject.issues?`· 异常 ${subject.issues} 帧 `:'';
  const yellow=document.createElement('span');yellow.className='yellow';yellow.textContent=subject.unverified?`· 待确认 ${subject.unverified} 帧 `:'';
  const legacy=document.createElement('span');legacy.style.color='#66adff';legacy.textContent=subject.recordings.some(r=>r.old_aoi_present)?'· 蓝色：旧软件已标注':'';
  counts.append(text,red,yellow,legacy);for(const r of subject.recordings)if(r.error){const error=document.createElement('span');error.textContent=' · '+r.error;counts.append(error)}section.append(counts);
  const track=document.createElement('div');track.className='subject-track';track.setAttribute('aria-label',named+'合并后的12个切片');
  for(const cell of subject.segments){
   const slot=document.createElement('div');slot.className='subject-slot';
   const name=document.createElement('span');name.className='segment-name';name.textContent=cell.segment;slot.append(name);
   const group=document.createElement('div');group.className='segment-parts';
   if(cell.missing){group.classList.add('missing-segment');group.setAttribute('aria-label',`切片 ${cell.segment} 缺失，斜纹标识`);group.title=`${cell.segment} 缺少录像或切片时间`}
   for(const part of cell.parts){
    const recording=subject.recordings[part.recording_index];
    const button=document.createElement('button');button.className='segment-track';button.disabled=!!recording.error||!recording.available;button.style.flexGrow=Math.max(.01,part.end_s-part.start_s);button.style.flexBasis='0';
    const complete=part.total&&part.computed>=part.total;
    const percent=complete?100:part.total?Math.min(99,Math.floor(100*part.computed/part.total)):0;
    button.title=`${named} · 切片 ${part.segment} · ${clock(part.start_s)}–${clock(part.end_s)} · 已标 ${percent}% · 异常 ${part.issues} 帧 · 待确认 ${part.unverified} 帧`+(recording.error?' · '+recording.error:!recording.available?' · 硬盘未连接':'');
    if(recording.old_aoi_present)button.title=`${named} · 切片 ${part.segment} · 旧软件已标注（蓝色） · ${clock(part.start_s)}–${clock(part.end_s)}`+(recording.error?' · '+recording.error:!recording.available?' · 硬盘未连接':'');
    button.setAttribute('aria-label',button.title);
    if(recording.old_aoi_present){const bar=document.createElement('span');bar.style.left='0%';bar.style.width='100%';bar.style.background='#4b96eb';button.append(bar)}
    else for(const span of part.spans){const bar=document.createElement('span');bar.style.left=100*(span.start-part.first)/part.total+'%';bar.style.width=100*(span.end-span.start+1)/part.total+'%';bar.style.background=progressColor(span.status);button.append(bar)}
    button.onclick=event=>{
     const rect=button.getBoundingClientRect(),ratio=event.detail===0?0:Math.max(0,Math.min(.999999,(event.clientX-rect.left)/rect.width));
     const firstProblem=event.detail===0?(part.spans.find(s=>s.status==='issue')||part.spans.find(s=>s.status==='unverified')):null;
     const index=firstProblem?.start??(part.total?part.first+Math.floor(ratio*part.total):null);
     const span=firstProblem||part.spans.find(s=>s.start<=index&&s.end>=index);
     directoryLocation(recording,part,index,span);
    };group.append(button);
   }
   slot.append(group);track.append(slot);
  }
  section.append(track);return section;
 }));
 if(!data.subjects.length)$('subjectDirectory').textContent='还没有可用的被试录像，可在下方添加。';
}
async function loadSubjectDirectory(){
 if(directoryLoading)return;directoryLoading=true;
 try{const response=await fetch('/api/directory');const data=await response.json();if(!response.ok)throw Error(data.error||'读取失败');const markup=JSON.stringify(data);if(markup!==directoryMarkup){drawSubjectDirectory(data);directoryMarkup=markup}}
 catch(error){$('subjectDirectory').textContent='目录读取失败：'+error.message}
 finally{directoryLoading=false}
}
// Read-only while the directory is open; closing it stops directory refreshes.
setInterval(()=>{if($('projectDialog').open&&!projectWaiting)loadSubjectDirectory()},5000);
