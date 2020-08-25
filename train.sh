# this command will get the expected result of 95.xx match the full precision model
python main_jay.py --epochs 90 --lr 0.01 --bit 3 --dataset cifar10 --wd 1e-4
python main_jay.py --epochs 90 --lr 0.01 --bit 4 --dataset cifar10 --wd 1e-4
python main_jay.py --epochs 1 --lr 0.01 --bit 8 --dataset cifar10 --wd 1e-4
python main_jay.py --epochs 120 --lr 0.01 --bit 32 --dataset cifar10 --wd 1e-4

python main_jay.py --epochs 90 --lr 0.01 --bit 8 --dataset imagenet --dataset_size 50 --wd 1e-4 --data_root /home/com13/data/data --home_root /home/com13/
