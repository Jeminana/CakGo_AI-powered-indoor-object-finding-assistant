// Browser Korean TTS. Each client owns its cooldown; users must tap Enable first.
(function (root) {
  class CakGoVoice {
    constructor(options = {}) {
      this.synth = options.synth || root.speechSynthesis;
      this.makeUtterance = options.makeUtterance || (text => new root.SpeechSynthesisUtterance(text));
      this.now = options.now || (() => Date.now());
      this.onStatus = options.onStatus || (() => {});
      this.cooldownMs = options.cooldownMs ?? 4000;
      this.missingCooldownMs = options.missingCooldownMs ?? 10000;
      this.enabled = false;
      this.lastKey = null;
      this.lastAt = -Infinity;
      this.pending = null;
      this.resetConfirmation();
      this.lastMissingAt = new Map();
    }
    enable() {
      if (!this.synth) { this.onStatus('이 브라우저는 음성 안내를 지원하지 않습니다.'); return false; }
      this.enabled = true;
      this.resetConfirmation();
      return this.say('음성 안내를 시작합니다.', 'enabled');
    }
    stop() {
      this.enabled = false;
      this.pending = null;
      this.lastKey = null;
      this.lastAt = -Infinity;
      this.resetConfirmation();
      this.lastMissingAt.clear();
      if (this.synth) this.synth.cancel();
    }
    resetConfirmation() {
      this.candidateKey = null;
      this.candidateFrames = 0;
    }
    guide(result) {
      const position = result.detected ? result.position : null;
      const messages = { left: '찾는 물건이 화면 왼쪽에 보입니다.', center: '찾는 물건이 화면 가운데에 보입니다.', right: '찾는 물건이 화면 오른쪽에 보입니다.' };
      const text = messages[position] || '현재 카메라 화면에서 찾는 물건이 보이지 않습니다.';
      const missing = !messages[position];
      const key = `${result.target}:${missing ? 'missing' : position}`;
      // Text is immediate; only audio waits for two consecutive responses.
      this.onStatus(text);
      if (!this.enabled || !this.synth) { this.resetConfirmation(); return false; }
      this.candidateFrames = key === this.candidateKey ? Math.min(2, this.candidateFrames + 1) : 1;
      this.candidateKey = key;
      if (this.candidateFrames < 2) return false;
      // A found result/error cannot bypass the missing-message repeat limit.
      const lastMissing = this.lastMissingAt.get(result.target) ?? -Infinity;
      if (missing && this.now() - lastMissing < this.missingCooldownMs) return false;
      const spoken = this.say(text, key, missing ? this.missingCooldownMs : this.cooldownMs);
      if (spoken && missing) this.lastMissingAt.set(result.target, this.now());
      return spoken;
    }
    error(kind) {
      this.resetConfirmation(); // Failed requests break the consecutive-frame sequence.
      const text = kind === 'camera' ? '카메라 연결을 확인해 주세요.' : '서버 연결을 확인해 주세요. 잠시 후 다시 시도해 주세요.';
      return this.say(text, `error:${kind}`);
    }
    say(text, key, cooldownMs = this.cooldownMs) {
      this.onStatus(text); // Text remains available even when speech is unsupported or disabled.
      if (!this.enabled || !this.synth) return false;
      if (key === this.lastKey && this.now() - this.lastAt < cooldownMs) return false;
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
