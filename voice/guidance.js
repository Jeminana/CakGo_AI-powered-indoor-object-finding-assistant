// Browser Korean TTS. Each client owns its cooldown; users must tap Enable first.
(function (root) {
  class CakGoVoice {
    constructor(options = {}) {
      this.synth = options.synth || root.speechSynthesis;
      this.makeUtterance = options.makeUtterance || (text => new root.SpeechSynthesisUtterance(text));
      this.now = options.now || (() => Date.now());
      this.onStatus = options.onStatus || (() => {});
      this.cooldownMs = options.cooldownMs ?? 4000;
      this.enabled = false;
      this.lastKey = null;
      this.lastAt = -Infinity;
      this.pending = null;
    }
    enable() {
      if (!this.synth) { this.onStatus('이 브라우저는 음성 안내를 지원하지 않습니다.'); return false; }
      this.enabled = true;
      return this.say('음성 안내를 시작합니다.', 'enabled');
    }
    stop() {
      this.enabled = false;
      this.pending = null;
      this.lastKey = null;
      this.lastAt = -Infinity;
      if (this.synth) this.synth.cancel();
    }
    guide(result) {
      const position = result.detected ? result.position : null;
      const messages = { left: '찾는 물건이 화면 왼쪽에 보입니다.', center: '찾는 물건이 화면 가운데에 보입니다.', right: '찾는 물건이 화면 오른쪽에 보입니다.' };
      const text = messages[position] || '현재 카메라 화면에서 찾는 물건이 보이지 않습니다.';
      return this.say(text, `${result.target}:${position}`);
    }
    error(kind) {
      const text = kind === 'camera' ? '카메라 연결을 확인해 주세요.' : '서버 연결을 확인해 주세요. 잠시 후 다시 시도해 주세요.';
      return this.say(text, `error:${kind}`);
    }
    say(text, key) {
      this.onStatus(text); // Text remains available even when speech is unsupported or disabled.
      if (!this.enabled || !this.synth) return false;
      if (key === this.lastKey && this.now() - this.lastAt < this.cooldownMs) return false;
      if (this.pending || this.synth.speaking || this.synth.pending) return false;
      const utterance = this.makeUtterance(text);
      utterance.lang = 'ko-KR';
      const korean = this.synth.getVoices().find(v => v.lang.toLowerCase().startsWith('ko'));
      if (korean) utterance.voice = korean;
      utterance.rate = 1;
      this.pending = utterance;
      utterance.onend = () => { if (this.pending === utterance) this.pending = null; };
      utterance.onerror = () => {
        if (this.pending === utterance) this.pending = null;
        this.onStatus(`${text} (음성 재생 실패: 화면 안내를 확인해 주세요.)`);
      };
      this.lastKey = key;
      this.lastAt = this.now();
      try { this.synth.speak(utterance); }
      catch (_) { utterance.onerror(); return false; }
      return true;
    }
  }
  root.CakGoVoice = CakGoVoice;
  if (typeof module !== 'undefined') module.exports = { CakGoVoice };
})(typeof window !== 'undefined' ? window : globalThis);
