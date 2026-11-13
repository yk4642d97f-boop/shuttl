#!/usr/bin/env python
"""Real TrackNetV2 Test split (29 clips, corrected labels): hits.py vs the
independent direction-reversal pseudo-labels of handoff-hit-model.log (half
window 0.16 s, exclusion gap 0.03 s, min speed 80 px/s, cos <= 0.1, NMS 0.20 s),
no learned re-scorer. Real hitter side is not measurable (labels carry no identity).

  .venv/bin/python -m scripts.accuracy.real_check
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from app.services.tracking import hits  # noqa: E402


def main() -> None:
    import hit_rescorer_data as D                          # noqa: E402
    import train_hit_rescorer as T                         # noqa: E402
    hits.load_learned_hit_model(None)                      # re-scorer stays OFF
    r = T.evaluate_real(Path(D.DEFAULT_TEST_ROOT), 0.5, None)
    print(f"real 29 clips: TP {r['tp']} FP {r['fp']} FN {r['fn']} P {r['precision']:.4f} "
          f"R {r['recall']:.4f} F1 {r['f1']:.4f} (hits latency {r['latency_ms_mean']:.1f} ms/clip)")


if __name__ == "__main__":
    main()
