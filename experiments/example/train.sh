export PYTHONPATH=../../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2
torchrun ../../main.py --dataset_file custom --world_size $1
