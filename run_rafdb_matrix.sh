#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/DSRUL_code/.venv/bin/python3}"
RAF_PATH="${RAF_PATH:-/新数据集/raf-basic}"
PRETRAINED="${PRETRAINED:-/新数据集/ijba_res18_naive.pth.tar}"
RESULT_ROOT="${RESULT_ROOT:-/result/rafdb_noise0.1_ACD_validation}"
SEEDS="${SEEDS:-0 1 2}"
VARIANTS="${VARIANTS:-baseline preupdate oracle_soft}"
SKIP_TEST_EVAL="${SKIP_TEST_EVAL:-1}"

case "${SKIP_TEST_EVAL}" in
  1) test_eval_args=(--skip_test_eval) ;;
  0) test_eval_args=() ;;
  *) echo "SKIP_TEST_EVAL must be 0 or 1" >&2; exit 2 ;;
esac

for seed in ${SEEDS}; do
  for variant in ${VARIANTS}; do
    save_dir="${RESULT_ROOT}/${variant}_seed${seed}"
    if [[ -d "${save_dir}" ]] &&
       [[ -n "$(find "${save_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to reuse non-empty save_dir: ${save_dir}" >&2
      exit 2
    fi
    mkdir -p "${save_dir}"
    "${PYTHON_BIN}" -u src/train_rafdb.py \
      --raf_path "${RAF_PATH}" \
      --train_label_file 0.1noise_train.txt \
      --clean_label_file list_patition_label.txt \
      --pretrained "${PRETRAINED}" \
      --save_dir "${save_dir}" \
      --ccar_variant "${variant}" \
      --oracle_noisy_weight 0.5 \
      --val_ratio 0.1 \
      --split_seed 0 \
      --seed "${seed}" \
      --target_val_accuracy 1.1 \
      "${test_eval_args[@]}" \
      --batch_size 256 \
      --workers 1 \
      --epochs 100 \
      --optimizer adam \
      --lr 0.0002 \
      --weight_decay 0.0001 \
      --lr_scheduler exponential \
      --lr_decay_gamma 0.99 \
      --train_sampler shuffle \
      --drop_rate 0.2 \
      --rotation_degrees 0 \
      --ema_decay 0 \
      --beta 0.2 \
      --gamma 0.6 \
      --alpha 0.4 \
      --prototype_momentum 0.9 \
      --ccar_temperature 0.1 \
      --relabel_epoch 8 \
      --margin_1 0.5 \
      --out_dimension 64 \
      2>&1 | tee "${save_dir}/train.log"
  done
done
