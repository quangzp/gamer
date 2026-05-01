# Evaluation Playbook

## Relevant files

- `challenge_desc.md`
- `docs/scripts.md`
- `scripts/test_SMB_decoder.sh`
- `SeqRec/tasks/test_SMB_decoder.py`

## Current local sanity baselines

These are majority-class action baselines measured from the local dataset layout in this repo:
- `JobChallenge`: about `0.6018`
- `JobChallenge_test`: about `0.5739`

If `action_acc` is around `0.9`, it is not explainable by class imbalance alone.

## Read results correctly

The JSON written by `test_SMB_decoder.py` is a list of dicts:
- several per-behavior entries
- one merged entry with `eval_type = "Merged Behavior"`

Use the merged entry for headline comparison unless the user specifically asks for per-behavior analysis.

## Submission checklist

- `export_submission=1`
- `submission_topk=10`
- `apply_behavior_label=apply` unless the dataset semantics differ
- `uid_map_file` and `item_map_file` present or inferable
- exported CSV columns:
  `session_id`, `action`, `job_id`

## Common interpretation mistakes

- treating `valid_test` as held-out challenge test
- comparing different beam counts without noting it
- discussing `action_acc` without looking at `mrr@10`
- assuming action is predicted by a dedicated classification head
