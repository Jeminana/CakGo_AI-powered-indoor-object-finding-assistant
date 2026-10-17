// Execute the actual page script and shared voice module with fake browser devices.
// This validates wiring/lifecycle, not real-device pronunciation or permission policy.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
assert.match(html, /<script src="\/voice\/guidance\.js"><\/script>/);
const pageScript = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1];
const voiceScript = fs.readFileSync(path.join(__dirname, '../voice/guidance.js'), 'utf8');

function setup(withModule = true) {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      textContent:'', value:'keys', hidden:true, videoWidth:0, videoHeight:0,
      setAttribute(name, value) { this[name] = value; },
      replaceChildren(...children) { this.children = children; },
      classList:{add() {}},
    });
    return elements.get(id);
  };
  const utterances = [];
  let stopped = 0;
  let cancelled = 0;
  let getCamera = async () => ({getTracks:()=>[{stop() {stopped++;}}]});
  let fetchResult = async () => ({ok:true, headers:{get:()=> 'mock'}, json:async()=>({
    target:'keys', detected:true, position:'left', image_width:300, image_height:200,
    detections:[{label:'keys',confidence:.9,bbox:[0,0,60,100]}],message:'찾는 물건이 화면 왼쪽에 보입니다.',
  })});
  const events = {};
  const window = {
    speechSynthesis:{speak:u=>utterances.push(u), cancel(){cancelled++;},getVoices:()=>[{lang:'ko-KR'}]},
    SpeechSynthesisUtterance:function(text) {this.text = text;},
    addEventListener:(name, fn)=>{events[name]=fn;},
  };
  const context = vm.createContext({window, console,
    document:{getElementById:element, createElement:()=>({
      getContext:()=>({drawImage(){}}),toBlob:resolve=>resolve({}),
    }),createElementNS:()=>({setAttribute(){},classList:{add(){}}})},
    navigator:{mediaDevices:{getUserMedia:(...args)=>getCamera(...args)}},
    FormData:class {append(){}}, AbortController,
    fetch:(...args)=>fetchResult(...args),setTimeout:()=>1,clearTimeout(){},
  });
  if (withModule) vm.runInContext(voiceScript, context);
  vm.runInContext(pageScript, context);
  return {element, utterances, events, context,
    run:code=>vm.runInContext(code,context),
    finishSpeech:()=>{utterances.at(-1)?.onend();},
    camera:fn=>{getCamera=fn;},fetch:fn=>{fetchResult=fn;},
    stopped:()=>stopped,cancelled:()=>cancelled};
}

(async()=>{
  const page = setup();
  assert.equal(page.utterances.length,0,'no speech before user gesture');
  await page.element('startBtn').onclick();
  assert.match(page.utterances[0].text,/시작/);
  page.finishSpeech();
  await page.run('detectOnce(scanId)');
  assert.match(page.utterances.at(-1).text,/왼쪽/);
  assert.equal(page.element('result').textContent,'찾는 물건이 화면 왼쪽에 보입니다.');
  page.finishSpeech();
  const count = page.utterances.length;
  await page.run('detectOnce(scanId)');
  assert.equal(page.utterances.length,count,'repeated frames use shared cooldown');

  page.element('voiceBtn').onclick();
  await page.run('detectOnce(scanId)');
  assert.equal(page.utterances.length,count,'muted results remain silent');
  assert.equal(page.element('voiceBtn')['aria-pressed'],'false');
  page.element('voiceBtn').onclick();
  page.finishSpeech();
  await page.run('detectOnce(scanId)');
  assert.match(page.utterances.at(-1).text,/왼쪽/);

  page.element('targetObject').value='wallet';
  const cancellations = page.cancelled();
  page.element('targetObject').onchange();
  assert.ok(page.cancelled()>cancellations,'switching target cancels old speech');
  page.finishSpeech();
  page.fetch(async()=>({ok:true,headers:{get:()=> 'mock'},json:async()=>({
    target:'wallet',detected:false,position:null,image_width:300,image_height:200,
    detections:[],message:'현재 카메라 화면에서 찾는 물건이 보이지 않습니다.',
  })}));
  await page.run('detectOnce(scanId)');
  assert.match(page.utterances.at(-1).text,/보이지 않습니다/);
  page.finishSpeech();
  page.fetch(async()=>{throw new Error('offline');});
  await page.run('detectOnce(scanId)');
  assert.match(page.utterances.at(-1).text,/서버 연결/);

  // A late API response after Stop must not draw boxes or restart speech.
  let answer;
  page.fetch(()=>new Promise(resolve=>{answer=resolve;}));
  const pending = page.run('detectOnce(scanId)');
  await new Promise(resolve=>setImmediate(resolve));
  page.element('stopBtn').onclick();
  const stoppedCount=page.utterances.length;
  answer({ok:true,headers:{get:()=> 'mock'},json:async()=>({})});
  await pending;
  assert.equal(page.utterances.length,stoppedCount);
  assert.equal(page.element('result').textContent,'');
  assert.ok(page.stopped()>0);

  const permissionPage=setup();
  let grant;
  permissionPage.camera(()=>new Promise(resolve=>{grant=resolve;}));
  const opening=permissionPage.element('startBtn').onclick();
  permissionPage.element('stopBtn').onclick();
  let lateTrackStopped=false;
  grant({getTracks:()=>[{stop(){lateTrackStopped=true;}}]});
  await opening;
  assert.equal(lateTrackStopped,true,'Stop during permission prompt closes late stream');
  assert.equal(permissionPage.element('camera').srcObject,null);

  const noModule=setup(false);
  assert.equal(noModule.element('voiceBtn').disabled,true);
  await noModule.element('startBtn').onclick();
  assert.equal(noModule.element('status').textContent,'Camera is running.');
  noModule.events.pagehide();
  assert.equal(noModule.element('camera').srcObject,null);
  console.log('Frontend voice integration passed: activation, detection, cooldown, mute, target change, missing, errors, stale response, pending permission, module fallback, page exit.');
})().catch(error=>{console.error(error);process.exitCode=1;});
