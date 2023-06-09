import os
import argparse
import time
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import transforms
import gdown

# timm func
from timm.models import create_model
from timm.utils import AverageMeter, reduce_tensor, accuracy

from utils import random_seed, NormalizeByChannelMeanStd, distributed_init
from data.dataset import ImageNet
from model.resnet import resnet50, wide_resnet50_2
from model.resnet_denoise import get_FD
from model import vit_mae
from model.model_zoo import model_zoo

# def get_model(model_name):
#     backbone=model_zoo[model_name]['model']
#     ckpt_dir=model_zoo[model_name]['ckpt']
#     mean=model_zoo[model_name]['mean']
#     std=model_zoo[model_name]['std']
#     pretrained=model_zoo[model_name]['pretrained']
#     act_gelu=model_zoo[model_name]['act_gelu']
    
#     if backbone=='resnet50_rl':
#         model=resnet50()
#     elif backbone=='wide_resnet50_2_rl':
#         model=wide_resnet50_2()
#     elif backbone=='resnet152_fd':
#         model = get_FD()
#     elif backbone=='vit_base_patch16' or backbone=='vit_large_patch16':
#         model=vit_mae.__dict__[backbone](num_classes=1000, global_pool='')
#     else:
#         model_kwargs=dict({'num_classes': 1000})
#         if act_gelu:
#             model_kwargs['act_layer']=nn.GELU
#         model = create_model(backbone, pretrained=pretrained, **model_kwargs)
    
#     if not pretrained:
#         ckpt=torch.load(ckpt_dir, map_location='cpu')
#         model.load_state_dict(ckpt)

#     normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
#     model = torch.nn.Sequential(normalize, model)
#     return model

def get_model(model_name):
    backbone=model_zoo[model_name]['model']

    # ckpt_dir=model_zoo[model_name]['ckpt']
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
    parser.add_argument('--model_name', default='resnet50', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--num-classes', default=1000, type=int, help='number of classes')

    # data parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--crop-pct', default=0.875, type=float, metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', default=3, type=int, help='1: lanczos 2: bilinear 3: bicubic')
    parser.add_argument('--imagenet_val_path', default='/imagenet/val/', type=str, help='path to imagenet validation dataset')

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', default=True)

    # attack info
    parser.add_argument('--attack_types', type=str, nargs='*', default=('autoattack',), help='autoattack, pgd100')
    parser.add_argument('--norm', type=str, default='Linf', help='You can choose norm for aa attack', choices=['Linf', 'L2', 'L1'])
    
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
    
    # get model
    model = get_model(args.model_name).cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id], find_unused_parameters=True)

    # set dataloader
    # dataset_eval=ImageNet(root=args.imagenet_val_path, meta_file='./data/imagenet_val_1k.txt', transform=test_transform)
    args.imagenet_val_path='./src_data/ILSVRC2012_img_val'
    dataset_eval=ImageNet(root=args.imagenet_val_path, meta_file='./src_data/val.txt', transform=test_transform)
    sampler_eval=None
    if args.distributed:
        sampler_eval = torch.utils.data.distributed.DistributedSampler(dataset_eval)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset=dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler_eval,
        collate_fn=None,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    clean_acc=validate(model, dataloader_eval, args)
    print('Top1 acc of clean images is: {0:>7.4f}'.format(clean_acc['top1']))
    for eps_int in [4]:
        robust_acc=adv_validate(model, dataloader_eval, args, eps_int)
        print('Top1 acc of eps {0} is: {1:>7.4f}'.format(eps_int, robust_acc['top1']))


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


def adv_validate(model, loader, args, eps_int, log_suffix='robust acc'):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    eps=eps_int/255
    model.eval()

    # set attackers
    attackers={}
    for attack_type in args.attack_types:
        if attack_type == 'autoattack':
            if args.distributed:
                from adv.autoattack_ddp import AutoAttack
            else:
                from adv.autoattack import AutoAttack
            adversary = AutoAttack(model, norm=args.norm, eps=eps, version='standard')
            attackers[attack_type]=adversary
        elif attack_type == 'pgd100':
            from adv.adv_utils import pgd_attack
            attackers[attack_type]=pgd_attack
        elif attack_type == 'fgsm':
            from adv.adv_utils import fgsm_attack
            attackers[attack_type]=fgsm_attack

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        batch_size=target.size(0)
        robust_flag=torch.ones_like(target).cuda()

        # attack
        for attack_type in args.attack_types:
            if attack_type == 'autoattack':
                x_adv = attackers[attack_type].run_standard_evaluation(input, target, bs=target.size(0))
            elif attack_type == 'pgd100':
                x_adv = attackers[attack_type](input, target, model, eps, 100, 1/255, 1, gpu=args.device_id)
            elif attack_type == 'fgsm':
                x_adv = attackers[attack_type](input, target, model, 4/255, 4/255, gpu=args.device_id)

            with torch.no_grad():
                output = model(x_adv.detach())
                _, label=torch.max(output, dim=1)
                robust_label= label == target
                robust_flag = torch.logical_and(robust_flag, robust_label)
        
        acc1=robust_flag.float().sum(0) * 100. / batch_size

        if args.distributed:
            acc1 = reduce_tensor(acc1, args.world_size)
        torch.cuda.synchronize()

        top1_m.update(acc1.item(), output.size(0))
        batch_time_m.update(time.time() - end)
        end = time.time()

        log_name = 'Test ' + log_suffix + ' of eps ' + str(eps_int)
        print(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m))

    metrics = OrderedDict([('top1', top1_m.avg)])

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust test script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)