'use strict';
let projectCatalog=null,projectWaiting=false,analysisTimer=null,selectedImport=null;
const segmentIds=Array.from({length:6},(_,i)=>[`${i+1}.1`,`${i+1}.2`]).flat();
function projectMessage(t){$('projectMessage').textContent=t||''}
function clearImport(){selectedImport=null;$('importNotes').textContent='';$('createProject').textContent='保存并打开视频';$('createProject').disabled=false}
function fillTimes(rows=[]){
 $('sliceInputs').replaceChildren(...segmentIds.map(id=>{
  const tr=document.createElement('tr'),name=document.createElement('td');name.textContent=id;tr.append(name);
  for(const field of ['start_s','end_s']){const td=document.createElement('td'),input=document.createElement('input');input.dataset.segment=id;input.dataset.field=field;input.setAttribute('aria-label',id+(field==='start_s'?' 开始':' 结束'));input.value=rows.find(r=>String(r.segment_id)===id)?.[field]??'';input.style.width='130px';td.append(input);tr.append(td)}return tr;
 }));
}
function timingImages(){
 const subject=$('newSubject').value,count=projectCatalog.timing_images[subject]||0;
 $('timingImages').replaceChildren(...Array.from({length:count},(_,i)=>{const img=document.createElement('img');img.src=`/api/timing/${subject}/${i}`;img.alt=`被试${subject}切片时间原记录 ${i+1}`;img.style.width='100%';img.loading='lazy';return img}));
}
function parseCsv(text){
 const rows=[];let row=[],value='',quoted=false;
 for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){value+='"';i++}else quoted=!quoted}else if(c===','&&!quoted){row.push(value);value=''}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(value);if(row.some(v=>v.trim()))rows.push(row);row=[];value=''}else value+=c}
 if(quoted)throw Error('CSV引号不成对');row.push(value);if(row.some(v=>v.trim()))rows.push(row);if(!rows.length)throw Error('CSV为空');return rows;
}
function seconds(value){
 const text=value.trim();if(!text)throw Error('12 个切片的起止时间都需要填写');
 const parts=text.split(':');if(parts.length>3||parts.some(p=>!/^\d+(\.\d+)?$/.test(p)))throw Error('时间格式应为秒数或分:秒');
 return parts.reduce((sum,p)=>sum*60+Number(p),0);
}
async function projectAction(work){
 if(projectWaiting)return;projectWaiting=true;projectMessage('正在处理，请稍候；首次读取视频会建立时间索引。');
 for(const id of ['loadProject','createProject','importReview','runAnalysis'])$(id).disabled=true;
 try{await work()}catch(e){projectMessage(e.message)}finally{projectWaiting=false;for(const id of ['loadProject','createProject','importReview','runAnalysis'])$(id).disabled=false}
}
$('projectDialog').addEventListener('cancel',e=>{if(projectWaiting)e.preventDefault()});
$('openProject').onclick=async()=>{
 if(!await save())return;$('projectDialog').showModal();$('projectTools').open=false;loadSubjectDirectory();projectMessage('正在读取本地索引…');
 try{
  const r=await fetch('/api/catalog');if(!r.ok)throw Error('读取本地项目失败');projectCatalog=await r.json();
  $('savedProject').replaceChildren(...projectCatalog.projects.map(p=>{const o=document.createElement('option');o.value=p.id;o.textContent=p.label||`${p.subject}号 · ${p.kind} · ${p.id}`;return o}));$('savedProject').value=projectCatalog.current;
  $('indexedVideo').replaceChildren(new Option('选择被试与录像段',''),...projectCatalog.videos.map(v=>new Option(v.key?`【${v.annotation_status?.label||'进度待核对'}】${v.name}${v.available===false?' · 硬盘未连接':''}`:`${v.name} · ${v.path.split('/').at(-2)}`,v.key||v.path)));
  $('newSubject').replaceChildren(...projectCatalog.subjects.map(n=>new Option(n+'号',n)));
  $('projectCurrent').textContent='当前：'+(state.project?.video||'');clearImport();$('videoPath').value='';$('splitRecording').checked=false;$('addVideoDetails').open=false;fillTimes();timingImages();projectMessage('');pollAnalysis();
 }catch(e){projectMessage(e.message)}
};
$('newSubject').onchange=()=>{clearImport();$('indexedVideo').value='';fillTimes();timingImages()};
$('videoPath').addEventListener('input',()=>{clearImport();$('indexedVideo').value=''});
$('indexedVideo').onchange=()=>{
 clearImport();const video=projectCatalog.videos.find(v=>(v.key||v.path)===$('indexedVideo').value);if(!video)return;
 selectedImport=video.key?video:null;
 $('videoPath').value=video.path;$('newSubject').value=String(video.subject);
 fillTimes(video.manifest||(video.path===projectCatalog.preset25[0].video_path?projectCatalog.preset25:[]));timingImages();
 $('splitRecording').checked=Boolean(video.split_recording);
 const progress=video.annotation_status;
 const progressNote=progress?.old_aoi_present?'旧软件已标过AOI，优先保留，不重复全量标注；是否完整覆盖12片仍以复核记录为准。':progress?.code==='covered'?'本工具已覆盖此录像的已知切片；红色问题段仍需复核。':'';
 $('importNotes').textContent=(video.available===false?'未找到视频，请连接 Samsung_T5 硬盘。\n':'')+(progressNote?progressNote+'\n':'')+(video.notes||[]).join('\n');
 $('createProject').textContent=video.project_id?'打开已有标注':'保存并打开视频';$('createProject').disabled=video.available===false;
};
$('manifestFile').onchange=async()=>{
 try{
  const file=$('manifestFile').files[0];if(!file)return;const text=await file.text();let rows;
  if(file.name.endsWith('.json'))rows=JSON.parse(text);else{
   const lines=parseCsv(text);const header=lines.shift().map(s=>s.replace(/^\uFEFF/,'').trim());
   rows=lines.map(cols=>Object.fromEntries(header.map((h,i)=>[h,cols[i]])));
  }
  if(!Array.isArray(rows)||!rows.length||rows.some(r=>!segmentIds.includes(String(r.segment_id)))||new Set(rows.map(r=>String(r.segment_id))).size!==rows.length)throw Error('清单切片应属于 1.1–6.2，且不能重复；字段为 segment_id,start_s,end_s');
  clearImport();$('indexedVideo').value='';$('splitRecording').checked=rows.length<12;
  fillTimes(rows);projectMessage('已载入切片时间，请核对所选被试和视频。');
 }catch(e){projectMessage(e.message)}
};
async function switched(action,data){
 await stopBatch();const s=await safe(action,data);if(!s)throw Error($('feedback').textContent||'打开失败');
 $('projectDialog').close();
}
$('loadProject').onclick=()=>projectAction(()=>switched('project_open',{project_id:$('savedProject').value}));
$('createProject').onclick=()=>projectAction(async()=>{
 if(selectedImport?.project_id){await switched('project_open',{project_id:selectedImport.project_id});return}
 const split_recording=$('splitRecording').checked;
 const manifest=segmentIds.map(id=>{const r={segment_id:id};const values=['start_s','end_s'].map(field=>document.querySelector(`#sliceInputs input[data-segment="${id}"][data-field="${field}"]`).value);if(split_recording&&values.every(v=>!v.trim()))return null;for(const [i,field] of ['start_s','end_s'].entries())r[field]=seconds(values[i]);return r}).filter(Boolean);
 await switched('project_create',{subject:Number($('newSubject').value),video:$('videoPath').value.trim(),manifest,split_recording,import_key:selectedImport?.key});
});
$('importReview').onclick=()=>projectAction(async()=>{
 const file=$('reviewFile').files[0];if(!file)throw Error('请选择 .aoi.json 标注存档');if(file.size>63000000)throw Error('标注存档超过64MB限制');
 await switched('review_import',{package:JSON.parse(await file.text()),video:$('reviewVideo').value.trim()});
});
async function pollAnalysis(){
 clearTimeout(analysisTimer);
 try{const r=await fetch('/api/analysis'),s=await r.json();$('analysisStatus').textContent=(s.subject?s.subject+'号 · ':'')+{none:'',running:'计算中…',complete:'已完成',failed:'计算失败'}[s.status];$('analysisDownload').hidden=s.status!=='complete';if(s.status==='failed')projectMessage(s.error);if(s.status==='complete')projectMessage('结果已保存到：'+s.folder);if(s.status==='running')analysisTimer=setTimeout(pollAnalysis,2000)}catch(e){projectMessage(e.message)}
}
$('runAnalysis').onclick=()=>projectAction(async()=>{if(!await safe('analyze'))throw Error($('feedback').textContent);projectMessage('正在读取已保存的AOI快照并计算…');await pollAnalysis()});
