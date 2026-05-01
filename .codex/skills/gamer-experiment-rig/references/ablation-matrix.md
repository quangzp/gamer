# Ablation Matrix

## Stable baseline fields

Keep these fixed within one ablation table:
- dataset
- tasks
- test_task
- backbone
- index mode
- max history length
- beam count
- checkpoint selection
- seed

## Recommended experiment ladder

### Family 1: No-code comparisons

- `Qwen3Multi`
- `Qwen3MultiSoft`
- `Qwen3MultiGeq`
- `Qwen3MultiReverse`
- `Qwen3MultiDense`
- `Qwen3MultiRouterMoe`

### Family 2: Task and data shaping

- `smb_explicit`
- `smb_explicit_decoder_4`
- `smb_fixed_ratio_5_1_1`
- `smb_fixed_ratio_*` tuned to your behavior mix

### Family 3: Decoding

- `num_beams`: `10`, `20`, `40`
- `max_his_len`
- `temperature`

### Family 4: Code changes

- explicit action head
- score calibration between action and item generation
- new masking logic
- reranking on top of constrained generation

## Minimal report template

For each run keep:
- purpose
- exact train command
- exact test command
- checkpoint path
- results path
- merged metrics
- one-sentence conclusion
