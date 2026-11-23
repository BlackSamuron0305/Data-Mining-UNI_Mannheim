#!/bin/bash
# All training runs of the paper and their validation, for the stratified run and the balanced run (stratified_balanced).
# They read the tables of step 3: run uv run python -m src.pipeline first (or start_with: transformed).
# A step whose result is already there is skipped. Logs: data/04_data_mining/<selection>/logs/<step>.log
#
#   bash run_models.sh
#   PY=".venv/Scripts/python.exe" bash run_models.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-"uv run python"}
METHODS="naive_bayes logistic_regression svm decision_tree random_forest gradient_boosting xgboost neural_network"

step() {
  local name=$1
  local done=$2
  shift 2
  local runs="data/04_data_mining/$IE500_SELECTION"
  if [ -n "$done" ] && [ -e "$runs/$done" ]; then
    echo "$(date '+%F %T') skip $name ($IE500_SELECTION)"
    return
  fi
  mkdir -p "$runs/logs"
  echo "$(date '+%F %T') start $name ($IE500_SELECTION)"
  if ! "$@" > "$runs/logs/$name.log" 2>&1; then
    echo "$(date '+%F %T') FAILED $name, see $runs/logs/$name.log"
    exit 1
  fi
}

models() {
  local runs="data/04_data_mining/$IE500_SELECTION"
  step bucket_rate_month bucket_rate/month/run.json $PY -m src.04_data_mining.bucket_rate.train --source month
  step bucket_rate_train bucket_rate/train/run.json $PY -m src.04_data_mining.bucket_rate.train --source train
  step baselines baseline/.done sh -c "$PY -m src.04_data_mining.baseline.train && touch $runs/baseline/.done"
  for f in context engine; do
    cv=""
    if [ "$f" = engine ]; then
      cv="--cv 0"
    fi
    for m in $METHODS; do
      step "tune_${m}_$f" "$m/${f}_tuned/run.json" $PY -m src.04_data_mining.tune --method "$m" --features "$f" $cv "$@"
    done
  done
}

# the stratified run: tuning, the readable tree, ablation, validation, importance
export IE500_SELECTION=stratified
models
step tree_rules decision_tree/engine_depth4/run.json $PY -m src.04_data_mining.decision_tree.train --features engine --depths 4 --name engine_depth4
for f in context engine; do
  step "ablation_xgboost_$f" "ablation/.done_xgboost_$f" sh -c "$PY -m src.04_data_mining.ablation --method xgboost --features $f && mkdir -p data/04_data_mining/stratified/ablation && touch data/04_data_mining/stratified/ablation/.done_xgboost_$f"
done
step validate "" $PY -m src.05_evaluation.validate
step importance "" $PY -m src.04_data_mining.importance

# the balanced run: balanced training games with the settings of the stratified run
export IE500_SELECTION=stratified_balanced
models --params-from stratified --cv 0
export IE500_SELECTION=stratified
step validate_with_balanced "" $PY -m src.05_evaluation.validate --with stratified_balanced
echo "$(date '+%F %T') all done"
