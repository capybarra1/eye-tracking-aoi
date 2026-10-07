/* Changes only image appearance and future tracking, never AOI geometry. */
let imagePending=null,imageSending=false,imageProject=null;
function imageControls(){
 if(!state)return;
 const locked=want||state.running||state.batch?.active||pausing||!!drag||dirty||!!reviewUI||state.review?.status==='computing'||!!scrubContext||(typeof timingDraft!=='undefined'&&!!timingDraft)||(busy&&!imageSending);
 for(const id of ['imageBrightness','imageContrast','imageDefault'])$(id).disabled=!!locked;
 if(!imageSending&&!imagePending){const v=state.image_adjustments||{brightness:0,contrast:100};$('imageBrightness').value=v.brightness;$('imageContrast').value=v.contrast;}
 $('imageBrightnessValue').textContent=$('imageBrightness').value;
 $('imageContrastValue').textContent=$('imageContrast').value+'%';
 if(imageSending||imagePending){for(const id of ['toggle','reset','export','slice','seek','openProject','batchStart','batchSkip','reviewHead','reviewTail','reviewSave','editTiming','horizontalMode'])$(id).disabled=true;}
 $('imageNote').textContent=locked?'暂停标注后可调节亮度和对比度。':imageSending?'正在更新画面…':'自动记住当前录像的设置，同时用于预览和后续识别。';
}
async function changeImage(){
 imagePending={brightness:Number($('imageBrightness').value),contrast:Number($('imageContrast').value)};
 if(imageSending){imageControls();return;}
 imageProject=state.project?.id||'';imageSending=true;controls();
 try{
  while(imagePending){
   if((state.project?.id||'')!==imageProject){imagePending=null;break;}
   const settings=imagePending;imagePending=null;
   const result=await safe('image_adjustments',{project_id:imageProject,settings});
   if(!result){imagePending=null;break;}
  }
 }finally{imageSending=false;controls();}
}
$('imageBrightness').oninput=changeImage;$('imageContrast').oninput=changeImage;
$('imageDefault').onclick=()=>{$('imageBrightness').value=0;$('imageContrast').value=100;changeImage();};
