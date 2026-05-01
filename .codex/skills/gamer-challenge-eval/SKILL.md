---
name: gamer-challenge-eval
description: Evaluate GAMER on the `JobChallenge` datasets, interpret generative test results correctly, sanity-check `MRR` and `action_acc`, and export challenge-ready submission CSVs. Use when running `test_SMB_decoder`, reading `results/*.json`, checking whether a score is believable, or preparing a submission file.
---

# Gamer Challenge Eval

Evaluate `JobChallenge` runs with the same assumptions the repo actually uses.

## Start Here

Read:
- [references/eval-playbook.md](references/eval-playbook.md)
- `challenge_desc.md`
- `docs/scripts.md`
- `scripts/test_SMB_decoder.sh`
- `SeqRec/tasks/test_SMB_decoder.py`

Use the helper script when you only need a clean summary:
- `python .codex/skills/gamer-challenge-eval/scripts/summarize_results.py <results.json>`

## Workflow

When the user asks about evaluation, do this in order:
1. Identify whether the run is `valid`, `valid_test`, or `test`.
2. Identify the exact `test_task`, backbone, beam count, and checkpoint source.
3. Read the merged metrics in the JSON output, not just per-behavior lines in logs.
4. If `action_acc` looks unusually high, compare it against the majority-class baseline in the reference file before concluding anything.
5. If the run is for submission, verify CSV schema and mapping files before trusting the artifact.

## Key Interpretation Rules

Use these repo-specific rules:
- The challenge objective is weighted: `70% MRR + 30% action accuracy`.
- In the current generative branch, action is inferred from the winning behavior beam, not from a separate classifier head.
- A high `action_acc` alone is not enough. Always inspect merged `mrr@10` or the exact ranking metrics the user cares about.
- `smb_explicit_valid` is not the same as the held-out test-style evaluation; do not present it as leaderboard-equivalent.

## Submission Checks

Before trusting a submission file:
1. Confirm `export_submission=1`.
2. Confirm `submission_topk=10` unless the user intentionally changed it.
3. Confirm `apply_behavior_label` matches the dataset semantics.
4. Confirm `uid_map_file` and `item_map_file` are either provided or correctly inferred from the dataset directory.
5. Confirm exported rows contain `session_id`, `action`, and a JSON list in `job_id`.

## What To Avoid

Do not:
- claim a result is strong without saying whether it is `valid`, `valid_test`, or `test`
- compare runs with different beams or different `test_task` as if they were apples-to-apples
- quote only one per-behavior metric when the merged behavior score is what matters
- assume `action_acc` comes from an explicit action head unless the code has been changed
