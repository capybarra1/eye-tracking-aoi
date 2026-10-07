'use strict';
let workflowBusy=false,workflowPoll=false;
function workflowControls(){
 const batch=state?.batch||{},computing=state?.review?.status==='computing';
 $('batchStart').disabled=busy||want||workflowBusy||batch.active||computing;
 $('batchStart').textContent='自动标注';
 const current=batch.items?.[batch.position];
 $('batchSkip').disabled=busy||want||workflowBusy||!!reviewUI||computing||!['running','paused','retrying'].includes(batch.status)||!current||current.project_id!==state.project?.id||['complete','skipped'].includes(current.status);
 $('batchStatus').textContent=batch.total?`录像 ${batch.done}/${batch.total} · `+({running:'自动运行中',preparing:'正在准备下一录像',retrying:`自动回退重试 ${batch.retry?.attempt||1}/2`,paused:batch.needs_attention?'需要人工调整':'已暂停',complete:'已跑完',error:'已停止'}[batch.status]||'')+(batch.skipped?` · ${batch.skipped} 份已跳过`:'')+(batch.blocked?` · ${batch.blocked} 份受阻`:'')+(batch.error?' · '+batch.error:''):'';
 if(batch.scope_label)$('batchStatus').textContent+=` · ${batch.scope_label}`;
 $('skippedPanel').hidden=!batch.manual_skips?.length;
 $('skippedTitle').textContent=`已跳过（${batch.manual_skips?.length||0}）`;
 $('skippedList').replaceChildren(...(batch.manual_skips||[]).map(item=>{
  const li=document.createElement('li'),text=document.createElement('span'),button=document.createElement('button');
  text.textContent=`${item.label} · ${item.reason} `;button.textContent='放回队列';button.disabled=batch.active||busy||!!reviewUI;
  button.onclick=()=>safe('batch_restore',{project_id:item.project_id});li.append(text,button);return li;
 }));
 const locked=busy||want||workflowBusy||batch.active||computing||!!drag||!!pausing;
 $('reviewHead').disabled=locked||state.cursor>=state.last;
 $('reviewTail').disabled=locked||!reviewUI||state.cursor<=reviewUI.start;
 $('reviewHead').textContent=reviewUI?'重设首帧':'设首帧';
 $('reviewSave').disabled=locked||state.review?.status!=='ready'||reviewUI?.changed;
 $('reviewSave').hidden=!reviewUI;
 $('reviewClose').hidden=!reviewUI;$('reviewClose').disabled=busy||want||workflowBusy;
 $('reviewPlay').hidden=state.review?.status!=='ready';$('reviewPlay').disabled=locked;
 const note=$('reviewStatus');
 if(computing)note.textContent=`正在补齐中间变化 · ${state.review.progress}%`;
 else if(state.review?.status==='error')note.textContent=state.review.error;
 else if(reviewUI)note.textContent=`首帧 ${clock(reviewUI.headTime)}${reviewUI.tailTime!=null?' → 尾帧 '+clock(reviewUI.tailTime):' · 调整首帧后，拖到结尾调整并设尾帧'}`+(state.review?.status==='ready'?' · 已补齐，检查后保存':reviewUI.changed&&reviewUI.tail?' · 已调整，重新设尾帧补齐':'');
 else note.textContent='需要补齐一段时：先设首帧并调整，再拖到结尾调整、设尾帧。';
}

function reviewKey(s){return {polygons:{...(s.polygons.partition?{partition:{...s.polygons.partition}}:{}),screen:[structuredClone(s.screen_points)],tablet:structuredClone(s.polygons.tablet)},visible:{...s.visible}}}
function captureReviewKey(){
 if(!reviewUI||!dirty)return;
 reviewUI.edits[state.cursor]=payload();
 if(state.cursor===reviewUI.start)reviewUI.head=payload();
 if(state.cursor===reviewUI.end)reviewUI.tail=payload();
 reviewUI.changed=true;
}
function restoreReviewKey(s){
 if(!reviewUI||s.preview||want||s.running||s.batch?.active)return;
 const key=s.cursor===reviewUI.start?reviewUI.head:s.cursor===reviewUI.end?reviewUI.tail:reviewUI.edits[s.cursor];
 if(key){partition=key.polygons.partition?{...key.polygons.partition}:null;points=structuredClone(key.polygons.screen[0]);tablet=structuredClone(key.polygons.tablet);visible={...key.visible};screenDirty=true;$('screenVisible').checked=visible.screen;$('tabletVisible').checked=visible.tablet}
}
async function refreshWorkflow(){
 const r=await fetch('/api/state'),s=await r.json();if(!r.ok)throw Error(s.error);await render(s);return s;
}
async function stopBatch(){
 await pause();
 while(state.batch?.active){await new Promise(r=>setTimeout(r,400));await refreshWorkflow()}
}
async function detachKeyframePreview(){
 if(state.review?.status&&state.review.status!=='idle')return !!await safe('review_cancel');
 return true;
}
$('batchSkip').onclick=async()=>{if(!await detachKeyframePreview())return;await safe('batch_skip',{project_id:state.project.id,position:state.batch.position});controls()};
$('batchStart').onclick=async()=>{
 unlockAlertAudio();if(!await save()||!await detachKeyframePreview())return;await pause();await safe('batch_start');controls();
};
async function markReviewKey(which){
 if(busy||want||workflowBusy||state.batch?.active)return;
 const index=state.cursor;
 if(which==='head'){
  if(index>=state.last){message('请在切片结束前设置首帧');return}
  const key=payload();dirty=false;
  if(!await detachKeyframePreview())return;
  reviewUI={project:state.project?.id,segment:state.segment,start:index,end:null,head:key,tail:null,headTime:state.time_s,tailTime:null,edits:{},changed:true};
  message('首帧已记下。在主画面调整后，拖到结尾，再调整并点「设尾帧并补齐」。');
 }else{
  if(!reviewUI||index<=reviewUI.start){message('尾帧要晚于首帧');return}
  if(!await save())return;
  reviewUI.end=index;reviewUI.tail=payload();reviewUI.tailTime=state.time_s;reviewUI.changed=false;
  if(!await safe('review_preview',{method:'keyframes',start:reviewUI.start,end:reviewUI.end,head:reviewUI.head,tail:reviewUI.tail}))reviewUI.changed=true;
 }
 controls();draw();
}
$('reviewHead').onclick=()=>markReviewKey('head');
$('reviewTail').onclick=()=>markReviewKey('tail');
$('reviewPlay').onclick=()=>playReview();
$('reviewSave').onclick=async()=>{
 if(!reviewUI||dirty||reviewUI.changed){message('关键帧已调整，请重新设尾帧补齐后保存');return}
 if(await safe('review_commit')){await safe('review_cancel');reviewUI=null;dirty=false;await refreshWorkflow();message('已保存首尾之间；可继续正常标注。');controls()}
};
$('reviewClose').onclick=async()=>{
 if(busy||want)return;
 if(!await detachKeyframePreview())return;
 reviewUI=null;dirty=false;await refreshWorkflow();message('已取消本次关键帧草稿，原标注未改动。');controls();
};
async function playReview(){
 if(busy||want||!reviewUI||state.review?.status!=='ready')return;
 await save();const mine=++epoch;want=true;controls();
 if(state.cursor>=reviewUI.end||state.cursor<reviewUI.start)await safe('seek',{index:reviewUI.start});
 while(want&&mine===epoch&&reviewUI&&state.cursor<reviewUI.end){
  const begin=performance.now(),before=state.time_s,next=Math.min(reviewUI.end,state.cursor+Math.max(1,Math.round(playbackSpeed)));
  const s=await safe('seek',{index:next});if(!s)break;
  await new Promise(r=>setTimeout(r,Math.max(0,1000*(s.time_s-before)/playbackSpeed-(performance.now()-begin))));
 }
 want=false;controls();
}
setInterval(async()=>{
 if(workflowPoll||busy||want||drag||dirty||workflowBusy||!(state?.batch?.active||state?.review?.status==='computing'))return;
 workflowPoll=true;try{await refreshWorkflow()}catch(e){message(e.message)}finally{workflowPoll=false}
},900);
