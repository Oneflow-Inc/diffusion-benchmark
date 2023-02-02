set -e
export ONEFLOW_NNGRAPH_ENABLE_PROGRESS_BAR=1
time python3 examples/graph-cache/infer.py --save
time python3 examples/graph-cache/infer.py --load
