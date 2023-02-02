set -e
export ONEFLOW_NNGRAPH_ENABLE_PROGRESS_BAR=1
python3 examples/graph-cache/infer.py --save
python3 examples/graph-cache/infer.py --laod
