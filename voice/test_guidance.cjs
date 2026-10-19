const assert = require('node:assert/strict');
const {CakGoVoice} = require('./guidance.js');
const left = {target:'keys',detected:true,position:'left'};
const right = {...left,position:'right'};
const missing = {target:'keys',detected:false,position:null};

function setup(options = {}) {
  let clock = 0;
  let cancelled = 0;
  const spoken = [];
  const statuses = [];
  const synth = {speaking:false,pending:false,getVoices:()=>[{lang:'ko-KR'}],
    speak:u=>spoken.push(u),cancel:()=>{cancelled++;}};
  const voice = new CakGoVoice({synth,now:()=>clock,makeUtterance:text=>({text}),
    onStatus:text=>statuses.push(text),...options});
  return {voice,spoken,statuses,synth,time:n=>{clock=n;},cancelled:()=>cancelled,
    finish:()=>spoken.at(-1)?.onend(),enable(){voice.enable();this.finish();}};
}

// Muted/pre-enable frames never count; visible status does not wait for audio.
{
  const f = setup();
  assert.equal(f.voice.guide(left),false);
  assert.match(f.statuses.at(-1),/왼쪽/);
  assert.equal(f.spoken.length,0);
  f.enable();
  assert.equal(f.voice.guide(left),false);
  assert.equal(f.voice.guide(left),true);
  assert.equal(f.spoken.at(-1).lang,'ko-KR');
  assert.equal(f.voice.guide(left),false,'no overlapping speech');
  f.finish(); f.time(3999);
  assert.equal(f.voice.guide(left),false);
  f.time(4000);
  assert.equal(f.voice.guide(left),true,'found repeats retain four seconds');
  f.finish();
  assert.equal(f.voice.guide(right),false,'new direction needs two frames');
  assert.equal(f.voice.guide(right),true);
}

// Flickering direction, presence and target cannot combine into a confirmation.
{
  const f = setup(); f.enable();
  for (const result of [left,right,left,missing,left,missing,right,left]) {
    assert.equal(f.voice.guide(result),false);
  }
  const wallet = {...left,target:'wallet'};
  assert.equal(f.voice.guide(wallet),false);
  assert.equal(f.voice.guide(left),false);
  assert.equal(f.voice.guide(wallet),false);
  assert.equal(f.voice.guide(wallet),true);
}

// Missing reminders wait ten seconds, even with an intervening found result.
{
  const f = setup(); f.enable();
  assert.equal(f.voice.guide(missing),false);
  assert.equal(f.voice.guide(missing),true);
  assert.match(f.spoken.at(-1).text,/보이지 않습니다/); f.finish();
  f.time(4000);
  assert.equal(f.voice.guide(missing),false);
  assert.equal(f.voice.guide(left),false);
  assert.equal(f.voice.guide(left),true); f.finish();
  assert.equal(f.voice.guide(missing),false);
  assert.equal(f.voice.guide(missing),false,'found cannot reset missing cooldown');
  f.time(9999);
  assert.equal(f.voice.guide(missing),false);
  f.time(10000);
  assert.equal(f.voice.guide(missing),true,'exact ten-second boundary'); f.finish();
  f.time(19999);
  assert.equal(f.voice.guide(missing),false);
  f.time(20000);
  assert.equal(f.voice.guide(missing),true);
}

// Errors are immediate but break consecutive detection confirmation.
{
  const f = setup(); f.enable();
  assert.equal(f.voice.guide(left),false);
  assert.equal(f.voice.error('network'),true);
  assert.match(f.spoken.at(-1).text,/서버 연결/); f.finish();
  assert.equal(f.voice.guide(left),false);
  assert.equal(f.voice.guide(left),true); f.finish();
  assert.equal(f.voice.error('camera'),true);
  assert.match(f.spoken.at(-1).text,/카메라 연결/);
}

// Busy synthesis never queues stale answers or consumes a missing reminder.
{
  const f = setup(); f.enable();
  f.synth.speaking = true;
  assert.equal(f.voice.guide(missing),false);
  assert.equal(f.voice.guide(missing),false);
  f.synth.speaking = false; f.synth.pending = true;
  assert.equal(f.voice.guide(missing),false);
  f.synth.pending = false;
  assert.equal(f.voice.guide(left),false);
  assert.equal(f.voice.guide(left),true); f.finish();
  assert.equal(f.voice.guide(missing),false);
  assert.equal(f.voice.guide(missing),true);
}

// Stop/re-enable clears history and confirmation, including muted frames.
{
  const f = setup(); f.enable(); f.voice.guide(left); f.voice.stop();
  assert.equal(f.cancelled(),1);
  assert.equal(f.voice.guide(left),false); f.enable();
  assert.equal(f.voice.guide(left),false);
  assert.equal(f.voice.guide(left),true); f.finish();
  f.voice.stop(); f.enable();
  assert.equal(f.voice.guide(missing),false);
  assert.equal(f.voice.guide(missing),true); f.finish();
  f.voice.stop(); f.enable();
  assert.equal(f.voice.guide(missing),false);
  assert.equal(f.voice.guide(missing),true,'new scan starts fresh missing history');
}

console.log('Voice checks passed: two-frame confirmation, flicker/target reset, 10-second missing reminders, overlap, errors, opt-in and restart.');
