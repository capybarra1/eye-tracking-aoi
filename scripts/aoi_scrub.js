/* One request at a time; keep only the newest pending position. */
(function(root){
 function create({load,show,error,interval=75}){
  let generation=0,pending=null,running=false,timer=null,last=0;
  async function pump(){
   timer=null;if(running||pending===null)return;
   const item=pending,mine=generation;pending=null;running=true;last=Date.now();
   try{const result=await load(item);if(mine===generation)await show(result,()=>mine===generation)}
   catch(e){if(mine===generation)error(e)}
   finally{running=false;schedule()}
  }
  function schedule(){if(!running&&pending!==null&&timer===null)timer=setTimeout(pump,Math.max(0,interval-(Date.now()-last)))}
  return {push(item){pending=item;schedule()},cancel(){generation++;pending=null;if(timer!==null)clearTimeout(timer);timer=null}};
 }
 if(typeof module!=='undefined')module.exports={create};else root.AOIScrub={create};
})(typeof window!=='undefined'?window:globalThis);
