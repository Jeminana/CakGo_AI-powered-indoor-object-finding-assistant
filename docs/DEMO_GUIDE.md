# Demo guide: running CakGo live

How to run the CakGo demo: a phone finds a wallet, keys, glasses or a remote control and says in Korean where it
is. This guide replaces the manual commands in [`2026-10-17/DEMO_GUIDE.md`](2026-10-17/DEMO_GUIDE.md), which was
written before the one-command start script.

**Short version:** on the demo laptop, double-click `scripts\start_demo.cmd`, open the printed link on the
phone, choose an object, press **Start Camera**.

## 빠른 시작 (발표자용)

1. 노트북에서 `scripts\start_demo.cmd` 더블클릭 → 초록색 상자에 휴대폰 링크가 나올 때까지 대기 (처음에는 30~60초)
2. 링크를 휴대폰으로 보내서 **Safari 또는 Chrome**에서 열기 (카카오톡 안의 브라우저 사용 금지)
3. 물건 선택 → **Start Camera** → 카메라 허용 → "음성 안내를 시작합니다"가 들리면 준비 완료
4. 끝나면 노트북 창에서 **Enter** → 모든 서버와 링크가 꺼짐

---

## 1. What you need

| Item | Requirement |
|---|---|
| Demo laptop | Windows, **NVIDIA GPU** (tested: RTX 4050 Laptop). Without a GPU, Grounding DINO takes about 15 s per frame, which is too slow for a live demo. |
| Internet | On the laptop (for the HTTPS link) and on the phone (Wi-Fi or mobile data; they do not need the same network). |
| Phone | iPhone (Safari) or Android (Chrome). Tested on iPhone 15 Pro. |
| Objects | At least one of: wallet, keys, glasses/sunglasses, remote control (TV or air conditioner), plus a spot where none of them is visible. |

## 2. First-time setup on the demo laptop (once)

From the repository root, in PowerShell:

```powershell
git clone https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant.git
cd CakGo_AI-powered-indoor-object-finding-assistant
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build of PyTorch
.venv\Scripts\python -m pip install -r backend/requirements.txt -r detection/experiments/grounding_dino/requirements.txt
winget install --id Cloudflare.cloudflared                                                     # HTTPS link for the phone
```

Then check the whole chain once. This starts everything, sends a test request through the phone link, and stops:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -Check
```

It should end with `[check] everything works`. The first run downloads the Grounding DINO model (about 690 MB),
which can take a few minutes. A teammate's laptop can be checked the same way as a backup.

## 3. Before the demo (about 10 minutes before)

1. Plug in the laptop. Turn off sleep (Settings → System → Power: "Never" while plugged in). Close games and other
   heavy apps, because they share the GPU.
2. **Phone:** volume up, iPhone silent mode off, screen auto-lock 5 minutes or more. Allow camera access for the
   browser (iPhone: Settings → Apps → Safari → Camera; Android: Settings → Apps → Chrome → Permissions).
   A Korean voice should be installed; it usually is when the phone language is Korean.
3. **Start the demo:** double-click `scripts\start_demo.cmd`. Wait for the green box:

   ```
   =============================================================
    Phone link (open in Safari / Chrome, not in KakaoTalk):
      https://<random-words>.trycloudflare.com
   ```

4. **Send the link to the phone** (e.g. KakaoTalk "나와의 채팅"), then **copy it into Safari or Chrome**. In-app
   browsers often block the camera or the voice. For a room, show the link as a QR code.
5. Open the page once, press **Start Camera**, point at an object, and check that boxes and voice work. Then leave
   the demo window open.

## 4. Using the app

| On the page | What it does |
|---|---|
| **What are you looking for?** | Choose one: Wallet · Keys · Glasses / Sunglasses · Remote control |
| **Start Camera** | Opens the rear camera and starts searching; says "음성 안내를 시작합니다" |
| **Stop Camera** | Stops the camera and the voice |
| **Voice guidance: on/off** | Mutes or unmutes the voice; the text and boxes stay |

- The phone sends about one photo per second. The object gets a yellow box, and the thickest box is the one the
  guidance is about.
- The voice says **"찾는 물건이 화면 왼쪽 / 가운데 / 오른쪽에 보입니다"**. It only speaks when two frames in a
  row agree, so it comes slightly after the box. **"현재 카메라 화면에서 찾는 물건이 보이지 않습니다"** is repeated
  at most every 10 seconds.
- A box usually appears about 2–3 s after the object comes into view (median 2.7 s in the team's test). Move the
  phone slowly and hold it still for a moment.

## 5. Suggested demo sequence (about 2 minutes)

1. **Glasses:** choose "Glasses / Sunglasses", press Start Camera, point at the glasses → box + "…가운데에 보입니다".
2. **Direction:** move the phone so the glasses are on the left, then on the right → the voice follows.
3. **Not visible:** point at an empty area → "…보이지 않습니다".
4. **False-alarm fix:** choose "Remote control", point at a computer mouse or keyboard first (no remote reported),
   then at the real remote → found.
5. Optional: keys or wallet, close and well lit.

In the team's real-photo test the model did well with the object about 0.5–1 m away in good light, also on cluttered
desks, and was weaker far away (2–3 m) and in dim light. Pick the demo spot accordingly.

## 6. Sharing the link with others (e.g. the lecturer)

- The link **only works while the demo window is open** and the laptop is on and online.
- Pressing Enter or closing the window **ends the link for good**. Starting again gives a **new** link, which you
  have to send again.
- Anyone with the link can use the laptop's detector while it runs, so send it only to the people who need it, and
  say until when it will be available.
- Camera frames are processed in memory and not stored or logged.
- For viewing later, send the GitHub repository and the backup video instead.

## 7. Stopping

Press **Enter** in the demo window, or close it. The backend, the frontend server and the tunnel all stop, and the
link no longer works.

## 8. Backup video

Record a screen video of the demo sequence on the phone beforehand (iPhone: Control Center → Screen Recording;
Android: Quick Settings → Screen recorder, with device audio). Keep it on the presentation laptop in case the
internet or the link fails.

## 9. Troubleshooting

| Problem | What to do |
|---|---|
| `[fail] .venv not found` | Do the first-time setup (section 2). |
| `[fail] port 8000 (or 5173) is already in use` | An older demo window is still open. Close it, or restart the laptop, and start again. |
| `[warn] no CUDA GPU` | The laptop has no usable NVIDIA GPU; detection will take about 15 s per frame. Use the GPU laptop. |
| `[fail] backend runs in 'mock' mode` | Start through `start_demo.cmd`, not by hand. |
| `[fail] tunnel did not start` / link not reachable | Check the laptop's internet. Wait a minute (a new link needs a few seconds before phones can reach it), then reload on the phone. If needed, close the window and start again; the link will change. |
| Phone shows "Cannot reach the server" | The demo window was closed, the laptop went to sleep, or the link changed. Restart and use the new link. |
| Camera does not open | Open the link in Safari/Chrome (not KakaoTalk), check the browser's camera permission, and make sure the link starts with `https://`. |
| No voice | Turn off silent mode and raise the volume, tap **Start Camera** again (phones only allow speech after a tap), and check "Voice guidance: on". |
| First answer is slow | Normal: the model warms up on the first request; the start script already sends one warm-up request. |
| Wrong object reported | Hold the phone still for a moment; look-alikes (mouse, keyboard, router) can still be mistaken for a remote control. |

Logs are in `%TEMP%\cakgo_demo\`. Other options: `start_demo.ps1 -Model yolo` (fine-tuned YOLO), `-Model mock` (no
model, fixed fake boxes), `-NoTunnel` (laptop only, http://localhost:5173). Details: [`scripts/README.md`](../scripts/README.md).
