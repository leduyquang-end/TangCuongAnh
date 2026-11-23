const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = process.cwd();
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
(async () => {
  const health = await fetch('http://127.0.0.1:7860/api/health').then(r => r.json());
  assert.equal(health.app, 'kind-enhance-local');
  const tab = await fetch('http://127.0.0.1:9225/json/new?http://localhost:7860', {method:'PUT'}).then(r => r.json());
  const socket = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve,reject) => {socket.onopen=resolve;socket.onerror=reject;});
  let sequence = 0; const pending = new Map(); const errors = [];
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
    if (message.id && pending.has(message.id)) {
      const {resolve,reject} = pending.get(message.id);pending.delete(message.id);
      message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
    }
  };
  function call(method,params={}) {return new Promise((resolve,reject) => {const id=++sequence;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});}
  async function evaluate(expression) {
    const result=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  async function until(expression,timeout=120000) {
    const end=Date.now()+timeout;
    while(Date.now()<end){if(await evaluate(expression))return;await delay(200);}
    throw new Error('Timed out: '+expression+'; '+await evaluate('document.body.innerText'));
  }
  await call('Runtime.enable');
  await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false});
  await until('typeof choose === "function" && !!document.getElementById("files")');
  let shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
  fs.writeFileSync(path.join(root,'outputs/checks/local_web_empty.png'),Buffer.from(shot.data,'base64'));
  async function selectFiles(selector,files) {
    const document=await call('DOM.getDocument');
    const input=await call('DOM.querySelector',{nodeId:document.root.nodeId,selector});
    await call('DOM.setFileInputFiles',{nodeId:input.nodeId,files});
  }
  await selectFiles('#files',[path.join(root,'assets/demo/anh_toi.png')]);
  await until('items.length === 1 && !document.getElementById("run").disabled');
  await evaluate('document.getElementById("run").click()');
  await until('!running && items[0]?.result && document.getElementById("after").naturalWidth > 0');
  const single=await evaluate('({status:document.getElementById("status").textContent,download:document.getElementById("download").download,width:document.getElementById("after").naturalWidth})');
  assert.equal(single.width,600);assert.ok(single.download.endsWith('.png'));
  const directory=path.join(root,'outputs/checks/folder_input');fs.mkdirSync(directory,{recursive:true});
  const manifest=JSON.parse(fs.readFileSync(path.join(root,'data/split-local.json'),'utf8'));
  manifest.splits.test.slice(0,2).forEach((row,index)=>fs.copyFileSync(path.join(manifest.source.data_root,row.low),path.join(directory,`anh_${index+1}.png`)));
  await selectFiles('#folder',[directory]);
  await until('items.length === 2');
  const names=await evaluate('items.map(x=>x.file.webkitRelativePath)');
  assert.ok(names.every(x=>x.includes('/')),'Directory selection should retain relative paths');
  await evaluate('document.getElementById("run").click()');
  await until('!running && items.length === 2 && items.every(x=>x.result) && document.getElementById("after").naturalWidth > 0');
  await evaluate('document.getElementById("fileList").value="0";document.getElementById("fileList").dispatchEvent(new Event("change"))');
  await until('active === 0 && document.getElementById("after").src.endsWith(items[0].result.output)');
  await until('document.getElementById("before").complete && document.getElementById("after").complete');
  shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
  fs.writeFileSync(path.join(root,'doc/3_san_pham/assets/giao_dien_web_local.png'),Buffer.from(shot.data,'base64'));
  const mobile=await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await delay(300);
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'),true,'Mobile layout should not overflow');
  assert.deepEqual(errors,[]);
  const proof={verified:true,browser:'Microsoft Edge (headless)',single_image_upload:true,directory_selection:true,directory_relative_paths:names,folder_images_processed:2,results_switching:true,PNG_download_link:true,browser_errors:errors,mobile_horizontal_overflow:false,screenshot:'doc/3_san_pham/assets/giao_dien_web_local.png',training_performed:false};
  fs.writeFileSync(path.join(root,'outputs/checks/local_web_browser.json'),JSON.stringify(proof,null,2)+'\n');
  console.log(JSON.stringify(proof));
  await call('Browser.close');socket.close();
})().catch(error=>{console.error(error);process.exitCode=1;});
