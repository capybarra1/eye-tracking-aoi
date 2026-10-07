'use strict';
let timingDraft=null,timingLoading=false,timingQueued=null,timingGeneration=0;
function timingControls(){
 $('editTiming').disabled=!state||busy||pausing||!!drag||!!reviewUI||state.review?.status==='computing';
 if(!timingDraft)return;
 for(const id of ['toggle','reset','export','slice','seek','horizontalMode','screenVisible','tabletVisible','openProject','batchStart','batchSkip','reviewHead','reviewTail','reviewPlay','reviewSave'])$(id).disabled=true;
 for(const id of ['timingSetStart','timingSetEnd'])$(id).disabled=busy||timingLoading||timingQueued!==null||!timingDraft.preview||Number($('timingSeek').value)!==timingDraft.preview.index;
 $('timingSave').disabled=busy||timingDraft.first>timingDraft.last;
 $('timingCancel').disabled=busy;
}
function timingSummary(){
 if(!timingDraft)return;
 $('timingStart').textContent=`开始：${clock(timingDraft.start)} · 第 ${timingDraft.first+1} 帧`;
 $('timingEnd').textContent=`结束：${clock(timingDraft.end)} · 含第 ${timingDraft.last+1} 帧`;
 const total=state.timing.last_frame+1,a=100*timingDraft.first/total,b=100*(timingDraft.last+1)/total;
 $('timingBand').style.background=`linear-gradient(to right,#334039 ${a}%,#a2e370 ${a}%,#a2e370 ${b}%,#334039 ${b}%)`;
 $('timingMessage').textContent=timingDraft.first>timingDraft.last?'开始在结束之后，请重新选择边界。':'绿色带为拟保留范围；拖动仅预览，保存后才生效。';
 timingControls();
}
async function timingPreview(index){
 if(!timingDraft)return;
 index=Math.max(0,Math.min(state.timing.last_frame,index));$('timingSeek').value=index;timingQueued=index;
 timingControls();if(timingLoading)return;
 timingLoading=true;const generation=timingGeneration;
 try{
  while(timingQueued!==null&&timingDraft&&generation===timingGeneration){
   const at=timingQueued;timingQueued=null;
   const q=new URLSearchParams({project:timingDraft.project,segment:timingDraft.segment,revision:timingDraft.revision,index:at});
   const response=await fetch('/api/timing-preview?'+q),result=await response.json();
   if(!response.ok)throw Error(result.error||'预览失败');
   if(!timingDraft||generation!==timingGeneration)return;
   if(timingQueued!==null)continue;
   const image=new Image();image.src=result.image;await image.decode();
   if(!timingDraft||generation!==timingGeneration||timingQueued!==null)continue;
   $('timingImage').src=result.image;timingDraft.preview=result;
   $('timingClock').textContent=`${clock(result.time_s)} · 第 ${result.index+1} 帧`;
  }
 }catch(e){if(timingDraft&&generation===timingGeneration)$('timingMessage').textContent=e.message}
 finally{timingLoading=false;timingControls();if(timingQueued!==null&&timingDraft)timingPreview(timingQueued)}
}
function closeTiming(){timingGeneration++;timingDraft=null;timingQueued=null;$('timingPanel').hidden=true;controls()}
$('editTiming').onclick=async()=>{
 if(timingDraft){closeTiming();return}
 if(!state||busy||reviewUI||drag)return;
 if(!await save())return;
 if(want||state.running||state.batch?.active)await pause();
 if(state.running||state.batch?.active)return;
 timingDraft={project:state.project?.id||'',segment:state.segment,revision:state.revision,first:state.first,last:state.last,start:state.start_s,end:state.end_s,preview:null};
 timingGeneration++;$('timingPanel').hidden=false;$('timingTitle').textContent=`调整切片 ${state.segment} 的范围`;
 $('timingSeek').max=state.timing.last_frame;$('timingImage').removeAttribute('src');timingSummary();controls();
 await timingPreview(state.cursor);
};
$('timingSeek').oninput=()=>timingPreview(Number($('timingSeek').value));
$('timingPrev').onclick=()=>timingPreview(Number($('timingSeek').value)-1);
$('timingNext').onclick=()=>timingPreview(Number($('timingSeek').value)+1);
$('timingStart').onclick=()=>timingPreview(timingDraft.first);
$('timingEnd').onclick=()=>timingPreview(timingDraft.last);
$('timingSetStart').onclick=()=>{const p=timingDraft?.preview;if(!p||timingLoading)return;timingDraft.first=p.index;timingDraft.start=p.time_s;timingSummary()};
$('timingSetEnd').onclick=()=>{const p=timingDraft?.preview;if(!p||timingLoading)return;timingDraft.last=p.index;timingDraft.end=p.end_s;timingSummary()};
$('timingOriginal').onclick=()=>{if(!timingDraft)return;Object.assign(timingDraft,{first:state.timing.original_first,last:state.timing.original_last,start:state.timing.original_start_s,end:state.timing.original_end_s});timingSummary();timingPreview(timingDraft.first)};
$('timingCancel').onclick=closeTiming;
$('timingSave').onclick=async()=>{
 if(!timingDraft||busy)return;
 const draft=timingDraft;
 const result=await safe('timing_adjust',{project_id:draft.project,segment:draft.segment,first:draft.first,last:draft.last});
 if(result){closeTiming();message('切片范围已更新，已定位到最早未标注的位置；原有标注保留。')}
 else if(state.project?.id===draft.project&&state.segment===draft.segment){draft.revision=state.revision;$('timingMessage').textContent=$('feedback').textContent}
};
