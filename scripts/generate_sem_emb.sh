#!/bin/bash
# This script generates item semantic embeddings for a specified dataset using the main.py script.
: ${dataset:=Beauty}
: ${gpu:=0}
: ${semantic_model:=llama-3.1}
: ${checkpoint:=}
: ${max_sent_len:=2048}
: ${data_type:=}

if [ -z "${data_type}" ]; then
  if [[ "${dataset}" == JobChallenge* ]]; then
    data_type="single"
  else
    data_type="SMB"
  fi
fi

if [ -z "${checkpoint}" ]; then
  if [ "${semantic_model}" = "multilingual-e5-base" ]; then
    checkpoint="intfloat/multilingual-e5-base"
  elif [ "${semantic_model}" = "llama-3.1" ]; then
    checkpoint="Meta-llama/Meta-Llama-3.1-8B"
  else
    checkpoint="${semantic_model}"
  fi
fi

python main.py SemEmb \
  --dataset ${dataset} \
  --root ./data \
  --gpu_id ${gpu} \
  --plm_name ${semantic_model} \
  --plm_checkpoint ${checkpoint} \
  --max_sent_len ${max_sent_len} \
  --data_type ${data_type} \
