---
name: gamer-experiment-rig
description: Run disciplined experiments and ablations for the GAMER repo. Use when planning or executing controlled comparisons across backbones, SMB tasks, mask variants, beam settings, tokenization modes, history lengths, or decoding strategies, especially for the generative `JobChallenge` workflow.
---

# Gamer Experiment Rig

Design experiments that stay comparable across this repo's many moving parts.

## Start Here

Read:
- [references/ablation-matrix.md](references/ablation-matrix.md)
- `scripts/train_SMB_decoder.sh`
- `scripts/test_SMB_decoder.sh`
- `SeqRec/tasks/train_SMB_decoder.py`
- `SeqRec/tasks/test_SMB_decoder.py`

## Comparison Discipline

Hold these fixed unless the experiment is explicitly about them:
- dataset
- task and test task
- backbone family
- tokenization / index mode
- beam count
- max history length
- checkpoint selection rule
- seed

Change one variable at a time. If more than one variable must change, state it as a new experiment family rather than an ablation.

## Good Default Generative Grid

Use this progression for `JobChallenge` unless the user asks for something broader:
1. Establish a stable baseline with `Qwen3Multi` or `Qwen3SessionMulti`.
2. Compare existing config variants before editing code:
   `Qwen3Multi`, `Qwen3MultiSoft`, `Qwen3MultiGeq`, `Qwen3MultiReverse`, `Qwen3MultiDense`, `Qwen3MultiRouterMoe`
3. Compare dataset/task variants:
   `smb_explicit`, `smb_explicit_decoder_4`, `smb_fixed_ratio_*`
4. Compare decoding settings:
   `num_beams`, temperature, and history length
5. Only then attempt architectural changes.

## Reporting Format

For every run, record:
- train command
- test command
- output checkpoint path
- results JSON path
- merged metrics
- any known incompatibility with prior runs

When asked to summarize experiments, group by experiment family, not by filesystem path.

## Common Failure Modes

Watch for these comparability traps:
- changing both task and backbone at once
- comparing `valid` and `test` numbers directly
- mixing original semantic IDs, RQ-VAE IDs, and chunked IDs in one table without labeling them
- changing beams and then attributing the gain to architecture
- editing scripts in a way that changes output directory conventions and silently breaks checkpoint reuse
