Based on: https://github.com/DonkeyShot21/essential-BYOL

Sample commands for running CIFAR100 on a single GPU setup:
```
python main.py \
    --gpus 1 \
    --dataset CIFAR100 \
    --batch_size 256 \
    --max_epochs 1000 \
    --arch resnet18 \
    --precision 16 \
    --comment wandb-comment
```
and multi-GPU setup:
```
python main.py \
    --gpus 2 \
    --distributed_backend ddp \
    --sync_batchnorm \
    --dataset CIFAR100 \
    --batch_size 256 \
    --max_epochs 1000 \
    --arch resnet18 \
    --precision 16 \
    --comment wandb-comment
```

# Logging
Logging is performed with [Wandb](https://wandb.ai/site), please create an account, and follow the configuration steps in the terminal. You can pass your username using `--entity`. Training and validation stats are logged at every epoch. If you want to completely disable logging use `--offline`.

# Contribute
Help is appreciated. Stuff that needs work:
- [ ] test ImageNet performance
- [ ] exclude bias and bn from LARS adaptation (see comments in the code)
