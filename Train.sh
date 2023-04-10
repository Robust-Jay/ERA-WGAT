##torchrun --nproc_per_node=2 Train_multi_gpus.py --configs configs/2D/loss/MSE.json
##torchrun --nproc_per_node=2 Train_multi_gpus.py --configs configs/2D/loss/Charbonnier.json
#
## python -m torch.distributed.launch --nproc_per_node=2 Train_multi_gpus.py --configs configs/2D/loss/Charbonnier.json
#
#
##python -m torch.distributed.launch --nproc_per_node=2 Train_multi_gpus.py --config configs/loss/MSE.json
##python -m torch.distributed.launch --nproc_per_node=2 Train_multi_gpus.py --config configs/loss/MSE_VGG/0.1.json
##python -m torch.distributed.launch --nproc_per_node=2 Train_multi_gpus.py --config configs/loss/MSE_VGG/0.01.json
## python -m torch.distributed.launch --nproc_per_node=2 Train_multi_gpus.py --config configs/loss/MSE_VGG/0.001.json
#python -m torch.distributed.launch --nproc_per_node=2 Train_multi_gpus.py --config configs/loss/MSE_VGG/0.0001.json
##python -m torch.distributed.launch --nproc_per_node=2 Train_multi_gpus.py --config configs/loss/MSE_VGG/0.00001.json

python -m torch.distributed.launch --nproc_per_node=2 --master_port 47769 Train_multi_gpus.py --config configs/layers/6.json
python -m torch.distributed.launch --nproc_per_node=2 --master_port 47769 Train_multi_gpus.py --config configs/layers/4.json
python -m torch.distributed.launch --nproc_per_node=2 --master_port 47769 Train_multi_gpus.py --config configs/layers/5.json
python -m torch.distributed.launch --nproc_per_node=2 --master_port 47769 Train_multi_gpus.py --config configs/layers/3.json
