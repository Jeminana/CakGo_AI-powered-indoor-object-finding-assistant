const assert = require('node:assert/strict');
const {CakGoVoice} = require('./guidance.js');
let clock = 0;
const spoken = [];
const synth = {speaking:false,pending:false,getVoices:()=>[{lang:'ko-KR'}],
  speak: u => spoken.push(u), cancel:()=>{}};
const voice = new CakGoVoice({synth, now:()=>clock, makeUtterance:text=>({text})});
const left = {target:'keys',detected:true,position:'left'};
assert.equal(voice.guide(left),false); // No automatic audio before user gesture.
voice.enable(); spoken.at(-1).onend();
assert.equal(voice.guide(left),true);
assert.equal(voice.guide(left),false); // No overlapping queue.
spoken.at(-1).onend();
assert.equal(voice.guide(left),false); // Cooldown after completion.
clock = 5000;
assert.equal(voice.guide(left),true);
spoken.at(-1).onend();
assert.equal(voice.guide({...left,position:'right'}),true);
spoken.at(-1).onend();
assert.equal(voice.guide({target:'keys',detected:false,position:null}),true);
assert.match(spoken.at(-1).text,/보이지 않습니다/);
voice.stop();
assert.equal(voice.guide(left),false);
console.log('Voice checks passed: opt-in, overlap, cooldown, direction, missing, stop');
