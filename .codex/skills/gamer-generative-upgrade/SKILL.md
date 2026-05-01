---
name: gamer-generative-upgrade
description: Improve the generative branch of the GAMER sequential recommendation project. Use when modifying or analyzing `Qwen3Multi`, `Qwen3SessionMulti`, `LlamaMulti`, `TIGER`, decoder-only SMB training, behavior/action conditioning, masking, routing, decoding, or submission-facing item generation for `JobChallenge`.
---

# Gamer Generative Upgrade

Improve the generative recommender without re-discovering the repo structure from scratch.

## Workflow

Read these files first:
- [references/hotspots.md](references/hotspots.md)
- `SeqRec/tasks/train_SMB_decoder.py`
- `SeqRec/tasks/test_SMB_decoder.py`

If the change is architecture-specific, then read only the relevant model file:
- `SeqRec/models/generative/Qwen3Multi/model.py`
- `SeqRec/models/generative/Qwen3SessionMulti/model.py`
- `SeqRec/models/generative/LlamaMulti/model.py`
- `SeqRec/models/generative/TIGER/model.py`

When changing data semantics, also read:
- `SeqRec/datasets/SMB_dataset.py`
- `SeqRec/datasets/loading_SMB.py`
- `SeqRec/datasets/collator.py`

## Guardrails

Preserve these invariants unless the user explicitly asks to break compatibility:
- Keep `scripts/train_SMB_decoder.sh` and `scripts/test_SMB_decoder.sh` runnable with existing env var patterns.
- Keep `test_SMB_decoder.py` able to export submission CSV with `session_id`, `action`, `job_id`.
- Keep `session_ids`, `extended_session_ids`, and `actions` aligned with tokenized sequence length for decoder-only backbones.
- Preserve candidate-constrained generation via trie / `prefix_allowed_tokens_fn`; do not switch to unconstrained free-form decoding.

## What The Current Branch Already Does

Assume the current repo is more advanced than a plain causal LM:
- `Qwen3Multi` already supports several cross-mask variants via config, including `level`, `causal`, `reverse`, `geq`, and `soft`.
- The repo already contains config variants such as `Qwen3MultiSoft`, `Qwen3MultiGeq`, `Qwen3MultiReverse`, `Qwen3MultiDense`, and `Qwen3MultiRouterMoe`.
- `action_acc` is not produced by a dedicated classifier head in the generative branch. At test time, the repo runs generation per behavior and keeps the behavior whose best beam score wins.

Prefer exploiting those existing knobs before inventing a new architecture family.

## Recommended Intervention Order

Use this order unless the user asks for a different research direction:
1. Tune existing config variants and SMB tasks first.
2. Improve loss shaping or decoding logic second.
3. Add lightweight heads or rerankers third.
4. Add new routing or masking mechanisms last.

High-value changes in this repo usually land in one of these buckets:
- Multi-task loss for item generation plus explicit action prediction.
- Better score composition at inference, for example separate action and item scores instead of raw top-beam score comparison.
- Cross-mask ablations on `Qwen3Multi*` and `Qwen3SessionMulti`.
- History rebalance through `smb_fixed_ratio_*` rather than ad hoc sampling.
- Controlled beam / temperature / max history studies instead of architecture churn.

## Validation

After any non-trivial change:
1. Run one local train or resume command on a small controlled setup.
2. Run `scripts/test_SMB_decoder.sh` on the same checkpoint.
3. Compare both `MRR` metrics and `action_acc`; do not optimize one blindly.
4. If the change touches decoding or export, inspect the submission CSV schema.

When reporting results, always state:
- backbone
- task and test task
- dataset
- tokenization / index mode
- beam count
- whether the number came from validation or test-style evaluation
