#---------------------------------------------------
# Imports
#---------------------------------------------------
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
#from torchviz import make_dot
#from matplotlib import pyplot as plt
#from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import time
import pdb
from self_models import *
import sys
import os
import shutil
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def find_threshold(batch_size=512, timesteps=2500, architecture='VGG16'):


    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    model.module.network_update(timesteps=timesteps, leak=1.0)
    pos=0
    thresholds=[]
    
    def find(layer):
        max_act=0
        # for name, m in model.named_parameters():
        #     print(name)
        #     print(m.device)
        print('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if output>max_act:
                    max_act = output.item()

                #print('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                if batch_idx==0:
                    thresholds.append(max_act)
                    print(' {}'.format(thresholds))
                    model.module.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                    break
    
    if architecture.lower().startswith('vgg'):              
        for l in model.module.features.named_children():
            if isinstance(l[1], nn.Conv2d):
                find(int(l[0]))
        
        for c in model.module.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                if (int(c[0]) == len(model.module.classifier) -1):
                    break
                else:
                    find(int(l[0])+int(c[0])+1)

    if architecture.lower().startswith('res'):
        for l in model.module.pre_process.named_children():
            if isinstance(l[1], nn.Conv2d):
                find(int(l[0]))
        
        pos = len(model.module.pre_process)

        for i in range(1,5):
            layer = model.module.layers[i]
            for index in range(len(layer)):
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l],nn.Conv2d):
                        pos = pos +1

        for c in model.module.classifier.named_children():
            if isinstance(c[1],nn.Linear):
                if (int(c[0])==len(model.module.classifier)-1):
                    break
                else:
                    find(int(c[0])+pos)

    print('\n ANN thresholds: {}'.format(thresholds))
    return thresholds

def train(epoch):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')
    
    model.train()
    local_time = datetime.datetime.now()  
    #current_time = start_time
    #model.module.network_init(update_interval)

    for batch_idx, (data, target) in enumerate(train_loader):
               
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data) 
        
        loss = F.cross_entropy(output, target)
        #make_dot(loss).view()
        loss.backward()
        optimizer.step()       
        #pdb.set_trace()
        #optimizer.load_state_dict(optimizer.state_dict())
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        
        for key, value in model.module.leak.items():
            # maximum of leak=1.0
            model.module.leak[key].data.clamp_(min=0.0, max=1.0)

        losses.update(loss.item(), data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))
                
        if (batch_idx+1) % train_acc_batches == 0:
            temp1 = []
            temp2 = []
            for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp1 = temp1+[round(value.item(),5)]
            for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp2 = temp2+[round(value.item(),5)]
            print('\n\nEpoch: {}, batch: {}, train_loss: {:.4f}, train_acc: {:.4f}, threshold: {}, leak: {}, timesteps: {}, time: {}'
                    .format(epoch,
                        batch_idx+1,
                        losses.avg,
                        top1.avg,
                        temp1,
                        temp2,
                        model.module.timesteps,
                        datetime.timedelta(seconds=(datetime.datetime.now() - local_time).seconds)
                        )
                    )
            local_time = datetime.datetime.now()

            # state = {
            #         'accuracy'              : max_accuracy,
            #         'epoch'                 : epoch,
            #         'state_dict'            : model.state_dict(),
            #         'optimizer'             : optimizer.state_dict(),
            #         'thresholds'            : temp1,
            #         'timesteps'             : timesteps,
            #         'leak'                  : temp2,
            #         'activation'            : activation
            #     }
            # try:
            #     os.mkdir('./trained_models/snn/')
            # except OSError:
            #     pass 
            # filename = './trained_models/snn/'+identifier+'.pth'
            # if not args.dont_save:
            #     torch.save(state,filename)

    print('\nEpoch: {}, train_loss: {:.4f}, train_acc: {:.4f}'
                    .format(epoch,
                        losses.avg,
                        top1.avg,
                        )
                    )
      
def test(epoch):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')
    
    if args.test_only:
        temp1 = []  
        temp2 = []  
        for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):    
            temp1 = temp1+[round(value.item(),2)]   
        for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))): 
            temp2 = temp2+[round(value.item(),2)]   
        print('\n Thresholds: {}, leak: {}'.format(temp1, temp2))

    with torch.no_grad():
        model.eval()
        global max_accuracy
        
        for batch_idx, (data, target) in enumerate(test_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output  = model(data) 
            #pdb.set_trace()
            loss    = F.cross_entropy(output,target)
            pred    = output.max(1,keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(),data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))
            
            if test_acc_every_batch:
                
                print('\n Images {}/{} Accuracy: {}/{}({:.4f})'
                    .format(
                    test_loader.batch_size*(batch_idx+1),
                    len(test_loader.dataset),
                    correct.item(),
                    data.size(0),
                    top1.avg
                    )
                )
        
        temp1 = []
        temp2 = []
        for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp1 = temp1+[value.item()]
        for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp2 = temp2+[value.item()]
        
        if epoch>5 and top1.avg<0.15:
            print('\n Quitting as the training is not progressing')
            exit(0)

        if top1.avg>max_accuracy:
            max_accuracy = top1.avg
             
            state = {
                    'accuracy'              : max_accuracy,
                    'epoch'                 : epoch,
                    'state_dict'            : model.state_dict(),
                    'optimizer'             : optimizer.state_dict(),
                    'thresholds'            : temp1,
                    'timesteps'             : timesteps,
                    'leak'                  : temp2,
                    'activation'            : activation
                }
            try:
                os.mkdir('./trained_models/snn/')
            except OSError:
                pass 
            filename = './trained_models/snn/'+identifier+'.pth'
            if not args.dont_save:
                torch.save(state,filename)    
        
            #if is_best:
            #    shutil.copyfile(filename, 'best_'+filename)

        print(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f} time: {}'
            .format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100','IMAGENET'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG4','VGG5','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('-lr','--learning_rate',    default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=30,                 type=int,       help='number of training epochs')
    #parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    #parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--timesteps',              default=20,                 type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--scaling_factor',         default=0.3,                type=float,     help='scaling factor for thresholds at reduced timesteps')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear', 'STDB'])
    parser.add_argument('--alpha',                  default=0.3,                type=float,     help='parameter alpha for STDB')
    parser.add_argument('--beta',                   default=0.01,               type=float,     help='parameter beta for STDB')
    parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')
    parser.add_argument('--momentum',               default=0.95,               type=float,     help='momentum parameter for the SGD optimizer')
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--betas',                  default='0.9,0.999',        type=str,       help='betas for Adam optimizer'  )
    parser.add_argument('--dropout',                default=0.5,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch',   action='store_true',                        help='print acc of every batch during inference')
    parser.add_argument('--train_acc_batches',      default=1000,               type=int,       help='print training progress after this many batches')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
    parser.add_argument('--resume',                 default='',                 type=str,       help='resume training from this state')
    parser.add_argument('--dont_save',              action='store_true',                        help='don\'t save training model during testing')

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
           
    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = args.architecture
    learning_rate       = args.learning_rate
    pretrained_ann      = args.pretrained_ann
    pretrained_snn      = args.pretrained_snn
    epochs              = args.epochs
    #lr_reduce           = args.lr_reduce
    timesteps           = args.timesteps
    leak                = args.leak
    scaling_factor      = args.scaling_factor
    default_threshold   = args.default_threshold
    activation          = args.activation
    alpha               = args.alpha
    beta                = args.beta  
    optimizer           = args.optimizer
    weight_decay        = args.weight_decay
    momentum            = args.momentum
    amsgrad             = args.amsgrad
    beta1               = float(args.betas.split(',')[0])
    beta2               = float(args.betas.split(',')[1])
    dropout             = args.dropout
    kernel_size         = args.kernel_size
    test_acc_every_batch= args.test_acc_every_batch
    train_acc_batches   = args.train_acc_batches
    resume              = args.resume
    start_epoch         = 1
    max_accuracy        = 0.0
    
    log_file = './logs/snn/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 
    identifier = 'snn_'+architecture.lower()+'_'+dataset.lower()
    log_file+=identifier+'.log'
    
    if args.log:
        f = open(log_file, 'w', buffering=1)
        sys.stdout = f
    #else:
    #    f = sys.stdout

    if not pretrained_ann:
        ann_file = './trained_models/ann/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'
        if os.path.exists(ann_file):
            val = input('\n Do you want to use the pretrained ANN {}? Y or N: '.format(ann_file))
            if val.lower()=='y' or val.lower()=='yes':
                pretrained_ann = ann_file

    print('\n Run on time: {}'.format(datetime.datetime.now()))
    print(f'Process ID: {os.getpid()}')

    print('\n Arguments: ')
    for arg in vars(args):
        if arg == 'pretrained_ann':
            print('\t {:20} : {}'.format(arg, pretrained_ann))
        else:
            print('\t {:20} : {}'.format(arg, getattr(args,arg)))
    
    # Training settings
    
    #if torch.cuda.is_available() and args.gpu:
    #    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))
    elif dataset == 'IMAGENET':
        normalize   = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    #normalize       = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

    if dataset in ['CIFAR10', 'CIFAR100']:
        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])
        transform_test  = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR10':
        trainset    = datasets.CIFAR10(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 10
    
    elif dataset == 'CIFAR100':
        trainset    = datasets.CIFAR100(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 100
    
    elif dataset == 'MNIST':
        trainset   = datasets.MNIST(root='~/Datasets/mnist/', train=True, download=True, transform=transforms.ToTensor()
            )
        testset    = datasets.MNIST(root='~/Datasets/mnist/', train=False, download=True, transform=transforms.ToTensor())
        labels = 10

    elif dataset == 'IMAGENET':
        labels      = 1000
        traindir    = os.path.join('/local/a/imagenet/imagenet2012/', 'train')
        valdir      = os.path.join('/local/a/imagenet/imagenet2012/', 'val')
        trainset    = datasets.ImageFolder(
                            traindir,
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))
        testset     = datasets.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ])) 

    train_loader    = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader     = DataLoader(testset, batch_size=batch_size, shuffle=False)

    if architecture[0:3].lower() == 'vgg':
        model = VGG_SNN_STDB(vgg_name = architecture, activation = activation, labels=labels, timesteps=timesteps, leak=leak, default_threshold=default_threshold, dropout=dropout, kernel_size=kernel_size, dataset=dataset)
    
    elif architecture[0:3].lower() == 'res':
        model = RESNET_SNN_STDB(resnet_name = architecture, activation = activation, labels=labels, timesteps=timesteps,leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout, dataset=dataset)

    # if freeze_conv:
    #     for param in model.features.parameters():
    #         param.requires_grad = False
    
    #Please comment this line if you find key mismatch error and uncomment the DataParallel after the if block
    model = nn.DataParallel(model) 
    #model = nn.parallel.DistributedDataParallel(model)
    #pdb.set_trace()

    if pretrained_ann:
      
        state = torch.load(pretrained_ann, map_location='cpu')
        # cur_dict = model.state_dict()     
        # for key in state['state_dict'].keys():
        #     if key in cur_dict:
        #         if (state['state_dict'][key].shape == cur_dict[key].shape):
        #             cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
        #             print('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
        #         else:
        #             print('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
        #     else:
        #         print('\n Error: Loaded weight {} not present in current model'.format(key))
        # model.load_state_dict(cur_dict)

        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
        print('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))

        #If thresholds present in loaded ANN file
        if 'thresholds' in state.keys():
            thresholds = state['thresholds']
            print('\n Info: Thresholds loaded from trained ANN: {}'.format(thresholds))
            model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
        else:
            thresholds = find_threshold(batch_size=512, timesteps=100, architecture=architecture)
            model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
            
            #Save the threhsolds in the ANN file
            temp = {}
            for key,value in state.items():
                temp[key] = value
            temp['thresholds'] = thresholds
            torch.save(temp, pretrained_ann)
    
    elif pretrained_snn:
                
        state = torch.load(pretrained_snn, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
        print('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
        # cur_dict = model.state_dict()     
        # for key in state['state_dict'].keys():
        #     if key in cur_dict:
        #         if (state['state_dict'][key].shape == cur_dict[key].shape):
        #             cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
        #             print('\n Loaded {} from {}'.format(key, pretrained_snn))
        #         else:
        #             print('\n Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
        #     else:
        #         print('\n Loaded weight {} not present in current model'.format(state['state_dict'][key]))
        # model.load_state_dict(cur_dict)

        # if 'thresholds' in state.keys():
        #     if state['timesteps']!=timesteps or state['leak']!=leak:
        #         print('\n Timesteps/Leak mismatch between loaded SNN and current simulation timesteps/leak, current timesteps/leak {}/{}, loaded timesteps/leak {}/{}'.format(timesteps, leak, state['timesteps'], state['leak']))
        #     thresholds = state['thresholds']
        #     model.module.threshold_update(scaling_factor = 1.0, thresholds=thresholds[:])
        # else:
        #     print('\n Loaded SNN model does not have thresholds')

    print('\n {}'.format(model))
    
    #model = nn.DataParallel(model) 
    if torch.cuda.is_available() and args.gpu:
        model.cuda()

    #params = list(model.parameters())
    # for value in model.module.threshold.values():
    #     params += [value]

    # for value in model.module.leak.values():
    #     params += [value]
    
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay, betas=(beta1, beta2))
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=False)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, verbose=True)
    print('\n {}'.format(optimizer))
        
    # find_threshold() alters the timesteps and leak, restoring it here
    model.module.network_update(timesteps=timesteps, leak=leak)
    
    if resume:
        print('\n Resuming from checkpoint {}'.format(resume))
        state = torch.load(resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
        print('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
        
        # cur_dict = model.state_dict()     
        # for key in state['state_dict'].keys():
        #     if key in cur_dict:
        #         if (state['state_dict'][key].shape == cur_dict[key].shape):
        #             cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
        #             print('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
        #         else:
        #             print('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
        #     else:
        #         print('\n Error: Loaded weight {} not present in current model'.format(key))
        # model.load_state_dict(cur_dict)
        #pdb.set_trace()
        #thresholds = state['thresholds']
        #leak       = state['leak']
        #model.module.threshold_update(scaling_factor = 1.0, thresholds=thresholds[:])
        #model.module.network_update(timesteps=timesteps, leak=leak[:])
        
        epoch           = state['epoch']
        start_epoch     = epoch + 1
        max_accuracy    = state['accuracy']
        optimizer.load_state_dict(state['optimizer'])
        for param_group in optimizer.param_groups:
            learning_rate =  param_group['lr']

        print('\n Loaded from resume epoch: {}, accuracy: {:.4f} lr: {:.1e}'.format(epoch, max_accuracy, learning_rate))

    for epoch in range(start_epoch, epochs):
        start_time = datetime.datetime.now()
        if not args.test_only:
            train(epoch)
        test(epoch)
        scheduler.step()

    print('\n Highest accuracy: {:.4f}'.format(max_accuracy))




