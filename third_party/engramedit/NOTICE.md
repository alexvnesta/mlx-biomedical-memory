The LongCat additive memory replay in `engram_poc/longcat.py`, exact-key rules and
answer extraction in `engram_poc/longcat_keys.py`, and evaluation conventions in
`scripts/run_longcat.py` are adapted from:

- ModalityDance/EngramEdit, commit `fb4089be4eec2575847d7e71342ff545ccb70508`
- https://github.com/ModalityDance/EngramEdit
- Copyright (c) 2025 Modality Dance; MIT license reproduced in `LICENSE` here.

The preparation script downloads the upstream demonstration prompts at that
commit, and MQuAKE data from Princeton NLP at
`fb43dadc2d8cd19d08ce81c63d957b59deb3f3cd`. Their contents remain local. Released
edited vectors and LongCat weights are downloaded under their separate upstream
model licenses and are not committed to this repository.

The replay port preserves exact token-key additive residuals, EOS boundaries,
cross-chunk suffix history, and upstream answer extraction. Replaying an existing
edited checkpoint is distinct from learning new edits.

`engram_poc/longcat_editor.py` additionally adapts the target optimization and
joint regularized update allocation to a frozen 4-bit MLX backbone. It uses
explicit fixed expression templates, two contexts and local corpus frequency
calibration instead of the paper's full generated/Wikipedia resources. It is an
adaptation, not an exact paper replication. GradientSafeRouter separately adapts
MLX-LM code; attribution and license are in `third_party/mlx-lm`.
