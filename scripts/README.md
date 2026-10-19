# scripts/

## Start the demo: `start_demo.cmd` / `start_demo.ps1`

One command starts everything on the demo laptop: backend with the chosen model, `frontend/dev_server.py`,
and an HTTPS tunnel. It then prints the link to open on the phone.

| How | Command |
|---|---|
| Double-click | `scripts\start_demo.cmd` |
| Terminal (repo root) | `powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1` |
| Check a setup, then stop | `... start_demo.ps1 -Check` |
| Other model | `... start_demo.ps1 -Model yolo` or `-Model mock` |
| Without phone link | `... start_demo.ps1 -NoTunnel` (http://localhost:5173 on the laptop only) |

What it does:
1. Checks `.venv`, the GPU, `cloudflared` (`winget install --id Cloudflare.cloudflared`) and that ports
   8000 and 5173 are free.
2. Starts the backend. Default model: **Grounding DINO multi-prompt**, the team's choice after the real-photo
   test. It stops with an error if the backend comes up in mock mode by mistake.
3. Sends one test request to warm up the model, which takes 10–60 s the first time.
4. Starts the dev server and the tunnel, waits until the link works, and prints it.
5. **Press Enter (or close the window) to stop everything.** The tunnel closes, so the link stops working.

Logs: `%TEMP%\cakgo_demo\`. The link changes every time; open the new one on the phone in Safari or Chrome,
not in KakaoTalk's built-in browser. Anyone with the link can use the laptop's detector while it runs, so don't
post it publicly.

`demo_request.py` sends one grey test frame to `/api/detect`; the start script uses it for the warm-up and the
link check.
