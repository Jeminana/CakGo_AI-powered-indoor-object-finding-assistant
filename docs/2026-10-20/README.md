# CakGo documentation: final results (20 October 2026)

Prepared by **나즈미 / Jeminana**. This package updates the 17 October package
([`../2026-10-17/`](../2026-10-17/README.md)), which stays as the record of that date.

## Files

- [Result analysis: how the final model was chosen](RESULT_ANALYSIS.md)
- [**발표 자료 및 대본 (한국어)**: Korean slide text, speaker script and Q&A for the presentation](PRESENTATION_NOTES_KO.md)
- [Presentation plan and script (English reference): slide-by-slide changes to the 17 October deck](PRESENTATION_NOTES.md)
- [Progress since 17 October and remaining tasks](PROGRESS.md)
- Demo: [`scripts/README.md`](../../scripts/README.md) replaces the manual start commands in the 17 October
  `DEMO_GUIDE.md`.

## Conclusion

- **Final model: Grounding DINO tiny with the multi-prompt** (team decision, 20 October).
- Fine-tuned YOLO scored best on the 70 stock-style dataset images but found only 17–58 % of objects in 94 real
  phone photos.
- Grounding DINO's single-object prompt found almost everything but raised false alarms on 50–93 % of photos
  without the object.
- The multi-prompt (targets plus look-alike objects) gave the best balance: found 50–92 % with 3–21 % false
  alarms.
- Weak spots: far and dim scenes, and remote-control false alarms on button-like objects.
- **Main lesson:** test on real phone photos before choosing a model. The dataset benchmark pointed the wrong way.
