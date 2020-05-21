set -ex
CUDA_VISIBLE_DEVICES=0,1 python train_workshop_baseline.py --dataroot dataset/train_data --name workshop_baseline_notexture_tuning_v1 --model retrieval_workshop_baseline_tuning --dataset_mode retrieval_workshop_baseline --niter 15 --niter_decay 25 --crop_size 256 --fine_size 256 --num_threads 24 --lr 0.001 --batch_size 64  --gpu_ids 0,1 --continue_train

