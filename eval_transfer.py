import os
import sys
import argparse
import time
import gdown
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# timm func
from timm.models import create_model
from timm.utils import AverageMeter, reduce_tensor, accuracy


from utils import distributed_init, NormalizeByChannelMeanStd, random_seed
from data.dataset import ImageNet
from model.resnet import resnet50, wide_resnet50_2
from model.resnet_denoise import get_FD
from model import vit_mae
from model.model_zoo import model_zoo

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__name__))))

from ares_attack_torch import FGSM, MIM, DI2FGSM, TIFGSM, SI_NI_FGSM, VMI_fgsm

def generate_attacker(args, net):
    if args.attack_name == 'fgsm':
        attack = FGSM(net, eps=args.eps)
    elif args.attack_name == 'mim':
        attack = MIM(net, epsilon=args.eps, stepsize=args.stepsize, steps=args.steps, decay_factor=args.decay_factor)
    elif args.attack_name == 'dim':
        attack = DI2FGSM(net, eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, 
                            resize_rate=args.resize_rate, diversity_prob=args.diversity_prob)
    elif args.attack_name == 'tim':
        attack = TIFGSM(net, kernel_name=args.kernel_name, len_kernel=args.len_kernel, nsig=args.nsig, 
                            eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, resize_rate=args.resize_rate, 
                            diversity_prob=args.diversity_prob)  
    elif args.attack_name == 'si_ni_fgsm':
        #net, epsilon, scale_factor, stepsize, decay_factor, steps
        attack = SI_NI_FGSM(net, epsilon=args.eps, scale_factor=args.scale_factor,stepsize=args.stepsize, decay_factor=args.decay_factor, steps=args.steps)
    elif args.attack_name == 'vmi_fgsm':
        attack = VMI_fgsm(net, epsilon=args.eps, beta=args.beta, sample_number=args.sample_number,
                          stepsize=args.stepsize, steps=args.steps, decay_factor=args.decay_factor)  
    return attack


def get_model(model_name):
    backbone=model_zoo[model_name]['model']
    url = model_zoo[model_name]['url']

    src_path='./src_ckpt'
    ckpt_name=f'{model_name}_checkpoint.pth'
    ckpt_dir=os.path.join(src_path, ckpt_name)
    ckpt_list=os.listdir(src_path)
    if ckpt_name not in ckpt_list:
        gdown.download(url, ckpt_dir, quiet=False)
    
    mean=model_zoo[model_name]['mean']
    std=model_zoo[model_name]['std']
    pretrained=model_zoo[model_name]['pretrained']
    act_gelu=model_zoo[model_name]['act_gelu']
    
    if backbone=='resnet50_rl':
        model=resnet50()
    elif backbone=='wide_resnet50_2_rl':
        model=wide_resnet50_2()
    elif backbone=='resnet152_fd':
        model = get_FD()
    elif backbone=='vit_base_patch16' or backbone=='vit_large_patch16':
        model=vit_mae.__dict__[backbone](num_classes=1000, global_pool='')
    else:
        model_kwargs=dict({'num_classes': 1000})
        if act_gelu:
            model_kwargs['act_layer']=nn.GELU
        model = create_model(backbone, pretrained=pretrained, **model_kwargs)
    
    if not pretrained:
        ckpt=torch.load(ckpt_dir, map_location='cpu')
        model.load_state_dict(ckpt)

    normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
    model = torch.nn.Sequential(normalize, model)
    return model

def get_args_parser():
    parser = argparse.ArgumentParser('Robust training script', add_help=False)

    # local test parameters
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-backend', default='nccl', help='backend used to set up distributed training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Model parameters
    parser.add_argument('--model_names', type=str, nargs='*', default=('resnet50_normal',), help='models in model zoo')
    
    # data parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--crop-pct', default=0.875, type=float, metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', default=3, type=int, help='1: lanczos 2: bilinear 3: bicubic')

    #attack paremeters
    parser.add_argument('--attack_name', default='tim', type=str, help='Name of adversarial attack')
    parser.add_argument('--eps', type= float, default=8/255, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=8/255/18, help='linf: 8/2550.0 and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=20, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--decay_factor', type= float, default=1.0, help='momentum is used')
    parser.add_argument('--resize_rate', type= float, default=0.85, help='dim is used')    #0.9
    parser.add_argument('--diversity_prob', type= float, default=0.7, help='dim is used')    #0.5
    parser.add_argument('--kernel_name', default='gaussian', help= 'kernel_name for tim', choices= ['gaussian', 'linear', 'uniform'])
    parser.add_argument('--len_kernel', type= int, default=15, help='len_kernel for tim')
    parser.add_argument('--nsig', type= int, default=3, help='nsig for tim')
    parser.add_argument('--scale_factor', type= int, default=5, help='scale_factor for si_ni_fgsm, min 1, max 5')
    parser.add_argument('--beta', type= float, default=1.5, help='beta for vmi_fgsm')
    parser.add_argument('--sample_number', type= int, default=10, help='sample_number for vmi_fgsm')
    
    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', default=True)

    # evaluated datasets
    parser.add_argument('--imagenet_val_path', default='', type=str, help='path to imagenet validation dataset')
    
    parser.add_argument('--output_dir', default='./test_out', type=str, help='path to the output')
    
    return parser

def main(args):
    #distributed settings
    if "WORLD_SIZE" in os.environ:
        args.world_size=int(os.environ["WORLD_SIZE"])
    args.distributed=args.world_size>1
    distributed_init(args)

    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark = True

    # test transform without norm
    t = []
    if args.input_size > 32:
        size = int(args.input_size/args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=args.interpolation),
        )
        t.append(transforms.CenterCrop(args.input_size))
    else:
        t.append(
            transforms.Resize(args.input_size, interpolation=args.interpolation),
        )
    t.append(transforms.ToTensor())
    test_transform = transforms.Compose(t)
    
    # normal_models  adv_models
    model_names=args.model_names
    acc_np=torch.zeros([len(model_names), len(model_names)])
    for i, source_model in enumerate(model_names):
        print(f'Processing model {source_model}')
        model = get_model(source_model).cuda() 
    
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id], find_unused_parameters=True)

        # set dataloader
        args.imagenet_val_path='./src_data/ILSVRC2012_img_val'
        dataset_eval=ImageNet(root=args.imagenet_val_path, meta_file='./src_data/val.txt', transform=test_transform) # './imagenet_val_1k.txt'

        sampler_eval=None
        if args.distributed:
            sampler_eval = torch.utils.data.distributed.DistributedSampler(dataset_eval)
        dataloader_eval = torch.utils.data.DataLoader(
            dataset=dataset_eval,
            batch_size=model_zoo[source_model]['batch_size'],
            shuffle=False,
            num_workers=0,
            sampler=sampler_eval,
            collate_fn=None,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        clean_acc=validate(model, dataloader_eval, args)
        print('Top1 acc of clean images is: {0:>7.4f}'.format(clean_acc['top1']))
        adv_output=adv_generator(model, dataloader_eval, args)


        for j, transfer_model in enumerate(model_names):
            model = get_model(transfer_model).cuda()
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id], find_unused_parameters=True)
            metric=transfer_validate(model, transfer_model, adv_output, args)
            acc_np[i][j]=metric['top1']
    
    
    torch.distributed.barrier()
    if args.rank==0:
        np.save(os.path.join(args.output_dir, args.attack_name+'.npy'), acc_np)
        

def validate(model, loader, args, log_suffix='clean acc'):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(input)

        acc1, _ = accuracy(output, target, topk=(1, 5))

        if args.distributed:
            acc1 = reduce_tensor(acc1, args.world_size)

        torch.cuda.synchronize()

        top1_m.update(acc1.item(), input.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

        log_name = 'Test ' + log_suffix
        print(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m))

    metrics = OrderedDict([('top1', top1_m.avg)])

    return metrics

def transfer_validate(model, model_name, save_dict, args, log_suffix='transfer acc'):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    advs=save_dict['img']
    labels=save_dict['label']
    batch_size=model_zoo[model_name]['batch_size']
    
    batch_num=advs.shape[0]//batch_size
    for batch_idx in range(batch_num):
        input = advs[batch_idx*batch_size: (batch_idx+1)*batch_num].cuda()
        target = labels[batch_idx*batch_size: (batch_idx+1)*batch_num].cuda()
        with torch.no_grad():
            output = model(input)

        acc1, _ = accuracy(output, target, topk=(1, 5))

        if args.distributed:
            acc1 = reduce_tensor(acc1, args.world_size)

        torch.cuda.synchronize()

        top1_m.update(acc1.item(), input.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

    metrics = OrderedDict([('top1', top1_m.avg)])

    return metrics

def adv_generator(source_model, loader, args):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    source_model.eval()
    
    rank_tensor=[]
    rank_label=[]

    # set attackers
    attacker=generate_attacker(args, source_model)

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        batch_size=target.size(0)

        x_adv=attacker.forward(input, target)
        
        rank_tensor.append(x_adv)
        rank_label.append(target)
        
        with torch.no_grad():
            output = source_model(x_adv.detach())
            _, label=torch.max(output, dim=1)
            robust_label= label == target
        
        acc1=robust_label.float().sum(0) * 100. / batch_size

        if args.distributed:
            acc1 = reduce_tensor(acc1, args.world_size)

        torch.cuda.synchronize()

        top1_m.update(acc1.item(), output.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

        log_name = 'Test'
        print(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m))

    rank_tensor=torch.cat(rank_tensor, dim=0)
    rank_label=torch.cat(rank_label, dim=0)

    return {'img':rank_tensor, 'label':rank_label}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust test script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
