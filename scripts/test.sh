set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=/home/mcc/working/pixel_link/pylib/src:$PYTHONPATH
python test_pixel_link.py \
     --checkpoint_path=$2 \
     --dataset_dir=$3\
     --gpu_memory_fraction=-1
     

