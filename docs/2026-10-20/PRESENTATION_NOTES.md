# Presentation plan and script: final version

For whoever edits the deck. Snapshot: 20 October 2026. Presenter: to be decided.

**The presentation is in Korean:** slide text, speaker script and Q&A in Korean are in
[`PRESENTATION_NOTES_KO.md`](PRESENTATION_NOTES_KO.md). This English version is the reference for the written report
and has the same slide structure.

The 17 October deck ([`../2026-10-17/CakGo_Project_Update.pptx`](../2026-10-17/CakGo_Project_Update.pptx))
was made before any phone test. Its story ("fine-tuned YOLO looks promising") no longer holds. The plan below
reuses its design and slides 1–6 and 11, and replaces the rest. Numbers marked **[after phone test 2]** don't
exist yet; fill them in after the team's second phone test or drop that line.

## Slide map: 17 Oct deck → final deck

| Final | Title | Action |
|---|---|---|
| 1 | CakGo | **Update** date and presenters |
| 2 | Target users and scope | **Keep** (one wording change) |
| 3 | Camera to Korean guidance | **Update**: add the two-frame confirmation |
| 4 | What we built | **Update** (was "Implementation progress") |
| 5 | Step 1: dataset benchmark | **Update** (merges old slides 5 and 7) |
| 6 | Fine-tuned YOLO looked strong | **Update** (old slide 6 + caveat) |
| 7 | Phone test 1: a surprise | **New** |
| 8 | Fixing false alarms: multi-prompt | **New** |
| 9 | Real-photo test: 94 team photos | **New** (replaces old slide 9, the test plan) |
| 10 | Where it works and where it doesn't | **New** |
| 11 | Decision: Grounding DINO multi | **New** |
| 12 | Speed and hardware | **Update** (old slide 8) |
| 13 | Limitations and future work | **New** (replaces old slide 10, milestones) |
| 14 | Live demo | **New** |
| A1 | Appendix: training run | **Keep** (old slide 11) |
| A2 | Appendix: how we kept the test fair | **New** |

## Slides

### 1. CakGo
**On slide:** CakGo · Finding everyday objects with a phone camera and Korean voice guidance · Final results ·
20 October 2026 · Team: 나즈미, 김철연, 오가암, 상싱위

**Say:** CakGo helps people find a wallet, keys, glasses or a remote control with their phone camera. The phone
says in Korean whether the object is on the left, in the center or on the right. Today we show what we built,
how we chose the detection model, and what we learned.

### 2. Target users and scope (keep)
Keep the 17 October slide. Change "TV remotes" to "remote controls (TV and air conditioner)". During phone
test 1 an air-conditioner remote was the real case.

### 3. Camera to Korean guidance (update)
**On slide (pipeline):** Phone camera → HTTPS link → FastAPI backend → Grounding DINO (multi-prompt) → left /
center / right → boxes on screen + Korean voice. Add a small note: *voice speaks only when two frames in a row
agree.*

**Say:** The phone sends about one frame per second. The backend asks the detector for the selected object and
turns the best box into a direction. To avoid jumpy announcements, the voice only speaks when two frames in a row
give the same answer. "Not visible" is repeated at most every 10 seconds.

### 4. What we built (update)
**On slide (table):**

| Part | Owner | Status |
|---|---|---|
| Smartphone camera UI, Grounding DINO hooked into the backend, larger text | 오가암 | Done (PR #4, #6, #15) |
| Grounding DINO benchmark + multi-prompt, camera ↔ backend connection, DINO real-photo run | 김철연 | Done (PR #2, #5, #10, #16) |
| Backend API, Korean voice, voice on the phone, two-frame voice, slides | 상싱위 | Done (PR #3, #8, #9, #12) |
| Dataset, YOLO fine-tuning, real-photo test, demo start script | 나즈미 | Done (PR #1, #7, #11, #13, #14, #17) |

**Say:** Every part is merged and works end to end on a real iPhone.

### 5. Step 1: dataset benchmark (update)
**On slide:** 70 stock-style images (Objects In House), 15 per object + 10 without any target.

| Model | Finds | Strength | Problem |
|---|---|---|---|
| Pretrained YOLO | remote only | fast, 0 false alarms | 3 of 4 objects unsupported |
| Grounding DINO | all 4 by text | high recall | false alarms on 4–9 of 10 empty images |

**Say:** Pretrained YOLO can't find wallets, keys or glasses at all. Grounding DINO can find everything but raises
many false alarms. So we did two things in parallel: fine-tune YOLO, and fix Grounding DINO's false alarms.

### 6. Fine-tuned YOLO looked strong (update)
Keep the 17 October table (AP50 wallet 0.81, keys 0.97, glasses 1.00, remote 1.00; 0 false alarms). Add one line:
**"…on images from the same source as its training data."**

**Say:** After training on 6,514 images, YOLO looked excellent on this test set. Wallets were weaker: the model
found 100 % of training wallets but only 63 % of new ones, which is overfitting. A second, longer training run
didn't fix it. The real question was how it behaves on a phone.

### 7. Phone test 1: a surprise (new)
**On slide:**
- iPhone 15 Pro, real desk, 19 October. The full chain worked: camera → backend → boxes → Korean voice.
- Fine-tuned YOLO: found **1 of 4** objects in real photos.
- Grounding DINO: found **4 of 4**, but called a mouse, a laptop keyboard and a router "remote control".
- Testers: voice too chatty, text too small → both fixed.

*Optional image: one phone screenshot with the mouse boxed as a remote. Only use screenshots whose owner agrees;
they show a private desk.*

**Say:** This changed our plan. The model that won on the dataset failed on the phone. The dataset images are
clean stock photos; a real desk is cluttered and the lighting is different. Grounding DINO saw the real objects,
but it gave the word "remote control" to anything with buttons.

### 8. Fixing false alarms: multi-prompt (new)
**On slide:**
- Before: ask only "remote control." → the model gives that word to the most similar thing.
- After: ask "wallet. keys. glasses. remote control. computer mouse. laptop. keyboard. pen. …" → each box goes to
  its best word. A mouse becomes "computer mouse", not "remote control".
- Validation images: false alarms on images without the object **53–94 % → 5–16 %**.
- Tuned on validation images, never on the test photos.

**Say:** The fix needs no training. We give the model the names of things that look similar, so it can say
"that's a mouse". This cut false alarms about tenfold. The cost is that some real wallets get labelled with
another word, so wallet recall drops.

### 9. Real-photo test: 94 team photos (new)
**On slide:** chart or table (found / false alarm):

| Object | Fine-tuned YOLO | **Grounding DINO multi** | Grounding DINO single |
|---|---|---|---|
| Wallet | 17 % / 1 % | **50 % / 11 %** | 96 % / 50 % |
| Keys | 30 % / 1 % | **50 % / 15 %** | 100 % / 85 % |
| Glasses | 46 % / 10 % | **50 % / 3 %** | 96 % / 53 % |
| Remote | 58 % / 1 % | **92 % / 21 %** | 100 % / 93 % |
| Balance (found − false alarm) | 0.34 | **0.48** | 0.28 |

Footnote: 94 photos by 3 team members; identical photos for all models (fingerprint); screenshots excluded;
nothing tuned on these photos.

**Say:** We photographed our own belongings in our own rooms. Single-prompt DINO "finds" everything because it
says yes to everything. YOLO is cautious but misses most objects. Multi-prompt DINO has the best balance.

### 10. Where it works and where it doesn't (new)
**On slide:**
- ✅ Clutter 15/16 · partly hidden 7/8 · easy 7/8
- ⚠️ Dim light 4/8 · **far away (2–3 m) 2/8**
- Big differences between photographers: 75 % vs 38 % of objects found

**Say:** The model is good at the realistic case, a messy desk with the object partly covered. It struggles when
the object is far away or the room is dark. In the app the user can move closer and the camera keeps trying, but
we have not measured that yet.

### 11. Decision: Grounding DINO multi (new)
**On slide:**
- **Chosen:** Grounding DINO tiny, multi-prompt, threshold 0.35
- Why: best balance on real photos; finds all four objects without training; few false alarms after the fix
- Cost: needs an NVIDIA GPU (about 0.3 s per frame); weaker on far and dim scenes
- Lesson: **the dataset test did not predict real use; only the real-photo test should decide the model**

### 12. Speed and hardware (update)
**On slide:**

| Measurement | Time |
|---|---|
| Grounding DINO multi, model only (RTX 4050 laptop) | 0.32 s |
| Through the HTTPS link (one test request) | about 0.45 s |
| Phone: pointing at the object → spoken direction | **[after phone test 2]** |
| Without a GPU | about 15 s per frame → not usable live |

**Say:** For comparison, YOLO takes 15 ms, but speed doesn't help if the object isn't found. The voice waits for
two matching frames, so the spoken answer comes a little after the box appears.

### 13. Limitations and future work (new)
**On slide:**
- Far and dim scenes; about half of wallets, keys and glasses missed in a single frame
- Remote-control false alarms on button-like objects (21 %)
- Small test set (94 photos, 3 people); no study with users with visual difficulties yet
- Needs a GPU laptop or server
- Future: measure multi-frame behaviour, test with target users, vibration or directional sound cues
  (vibration works on Android only, not on iPhone), more real training photos, Microsoft ORBIT dataset

### 14. Live demo (new)
**On slide:** a QR code for the phone link, created on the day; the link changes each time.

**Demo order:** start `scripts\start_demo.cmd` about 10 minutes before → open the link on the phone → choose
Glasses → show a found object with voice → move it left/right → point at an empty area ("not visible") → choose
Remote and point at a mouse (shows the false-alarm fix). Backup: screen recording.

### A1. Appendix: training run (keep old slide 11)

### A2. Appendix: how we kept the test fair (new)
- 70 test images and their near-duplicates removed from YOLO's training data (leakage guard)
- Look-alike list tuned on validation images, checked on test photos only once
- Same 94 photos for every model, proven by a fingerprint; private photos never committed
- Screenshots and web images excluded from the real-photo test

## Likely questions

**Why not YOLO? It's 20× faster.** On real phone photos it found only 17–58 % of the objects. Speed doesn't help
if the object isn't found.

**50 % found sounds low.** That's per single photo. The app keeps sending frames while the user moves the phone,
so the object gets several chances. We haven't measured that rate yet; it's our next test.

**Why not use the single prompt? It finds 96–100 %.** It also says "found" on 50–93 % of photos where the object
isn't there. For a user who can't see the screen, a wrong direction is worse than "not visible".

**Did you tune on the test photos?** No. The look-alike words were chosen on separate validation images, and the
real photos were scored once with the default thresholds.

**Does it run on the phone?** No. The phone only captures and speaks; detection runs on a laptop with an NVIDIA
GPU, reached through an HTTPS link.

**Can it vibrate?** On Android, yes, technically easy. iPhone browsers don't support vibration, so we'd use
directional sounds instead. That's future work.

## Source map

- Slides 5–6: `tests/results/model_comparison_report.md`, `detection/experiments/yolo/README.md` (Results so far)
- Slide 7: PR #10 and PR #11 descriptions (phone test of 19 October)
- Slide 8: `detection/experiments/grounding_dino/MULTI_PROMPT.md`
- Slides 9–11: `tests/results/real_photos/COMPARISON.md`
- Slide 12: `detection/experiments/grounding_dino/results/rtx4050_gdino-tiny_multi_box0.05/metrics.md`; tunnel
  time from `scripts/start_demo.ps1 -Check`
- Slide 14: `scripts/README.md`
- Full analysis: [`RESULT_ANALYSIS.md`](RESULT_ANALYSIS.md)
