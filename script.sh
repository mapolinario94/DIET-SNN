#!/bin/bash

python ann.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 --learning_rate 0.01 --epochs 300 --optimizer SGD --dropout 0.2 --devices 4 --log --seed 0

python ann.py --dataset CIFAR100 --batch_size 64 --architecture VGG16 --learning_rate 0.01 --epochs 300 --optimizer SGD --dropout 0.2 --devices 4 --log --seed 0 --pretrained_ann trained_models/ann/ann_vgg16_cifar10.pth

export CUDA_VISIBLE_DEVICES=0

python snn.py --dataset CIFAR100 --batch_size 64 --architecture VGG16 --learning_rate 1e-4 --optimizer 'Adam' --epochs 100 --timesteps 5 --scaling_factor 0.6 --weight_decay 0 --dropout 0.1 --train_acc_batches 500 --default_threshold 1.0 --pretrained_ann ./trained_models/ann/ann_vgg16_cifar100.pth --log --activation Linear --leak 1.0 --alpha 0.3 --beta 0.01

python snn.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 --learning_rate 1e-4 --optimizer 'Adam' --epochs 100 --timesteps 5 --scaling_factor 0.6 --weight_decay 0 --dropout 0.1 --train_acc_batches 500 --default_threshold 1.0 --pretrained_ann ./trained_models/ann/ann_vgg16_cifar10.pth --log --activation Linear --leak 1.0 --alpha 0.3 --beta 0.01

#Train ANN
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture RESNET12 --learning_rate 1e-3 --epochs 300 --optimizer Adam --dropout 0.2 --devices 2 --log

#Train SNN
#python snn.py --dataset CIFAR10 --batch_size 64 --architecture RESNET12 --learning_rate 1e-4 --optimizer 'Adam' --epochs 300 --timesteps 5 --scaling_factor 0.2 --weight_decay 0 --dropout 0.2 --train_acc_batches 500 --devices 2 --default_threshold 0.4 --pretrained_ann './trained_models/ann/ann_resnet12_cifar10_Fri Sep 10 18:02:00 2021.pth' --dont_save --log #--individual_thresh #--dont_save #--test_only --test_acc_every_batch 