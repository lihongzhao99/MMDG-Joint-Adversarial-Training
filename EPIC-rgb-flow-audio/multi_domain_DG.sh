python train_EPIC.py --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999
python train_EPIC.py --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999
python train_EPIC.py --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999

python train_EPIC.py --use_video --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999

python train_EPIC.py --use_video --use_flow -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_flow -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_flow -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999

python train_EPIC.py --use_video --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999
python train_EPIC.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath ../../dataset/EPIC_KITCHENS/ --seed=0 --gpu=1 --use_dsu --ema_beta=0.999
