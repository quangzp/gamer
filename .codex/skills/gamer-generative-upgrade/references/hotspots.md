# Generative Hotspots

## Primary files

- `SeqRec/tasks/train_SMB_decoder.py`
  Drives tokenizer loading, backbone selection, config mutation, collator choice, and `TrainingArguments`.
- `SeqRec/tasks/test_SMB_decoder.py`
  Drives constrained generation, per-behavior testing, score aggregation, `action_acc`, and submission export.
- `SeqRec/datasets/collator.py`
  Aligns `input_ids`, `labels`, `session_ids`, `extended_session_ids`, and `actions`.
- `SeqRec/datasets/SMB_dataset.py`
  Generates the SMB history strings and the aligned per-token metadata used by decoder-only models.

## Model hotspots

- `SeqRec/models/generative/Qwen3Multi/model.py`
  Best place for action-aware masking, loss shaping, and decoding-oriented score changes.
- `SeqRec/models/generative/Qwen3SessionMulti/model.py`
  Best place for session-aware masking changes that depend on within-session structure.
- `SeqRec/models/generative/Qwen3Multi/router.py`
  Best place for behavior and action token alignment issues.

## Existing repo features worth reusing

- Cross-mask variants are already supported through config:
  `level`, `causal`, `reverse`, `geq`, `soft`.
- Config variants already exist in `config/s2s-models/`:
  `Qwen3MultiSoft`, `Qwen3MultiGeq`, `Qwen3MultiReverse`, `Qwen3MultiDense`, `Qwen3MultiRouterMoe`.
- The SMB loader already supports rebalanced history through `smb_fixed_ratio_*`.

## Improvement ideas that fit this codebase

- Add explicit action logits on top of the decoder hidden state and combine them with item generation scores at inference.
- Replace raw winning-beam behavior selection with calibrated score composition.
- Add loss terms or callbacks instead of rewriting the entire model family.
- Prefer ablations through config and dataset tasks before deep architectural refactors.
