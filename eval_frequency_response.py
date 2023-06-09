import os
import argparse
import time
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import gdown
# timm func
from timm.models import create_model
from timm.utils import AverageMeter, reduce_tensor, accuracy

from utils import random_seed, NormalizeByChannelMeanStd, distributed_init
from data.dataset import ImageNet
from model.resnet import resnet50, wide_resnet50_2
from model.resnet_denoise import resnet152_fd
from model import vit_mae
from model.model_zoo import model_zoo

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
        model = resnet152_fd()
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

    # model names
    parser.add_argument('--model_names', type=str, nargs='*', default=('resnet50_normal',), help='models in model zoo')

    # data parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--crop-pct', default=0.875, type=float, metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', default=3, type=int, help='1: lanczos 2: bilinear 3: bicubic')

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', default=True)

    # evaluated datasets
    parser.add_argument('--imagenet_val_path', default='./src_data/ILSVRC2012_img_val', type=str, help='path to imagenet validation dataset')

    parser.add_argument('--output_dir', default='', type=str, help='path to the output')

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
    for source_model in args.model_names:
        print(f'Processing model {source_model}')
        model = get_model(source_model).cuda() 
    
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id], find_unused_parameters=True)

        # set dataloader
        dataset_eval=ImageNet(root=args.imagenet_val_path, meta_file='./scr_data/val.txt', transform=test_transform) # './imagenet_val_1k.txt'

        sampler_eval=None
        if args.distributed:
            sampler_eval = torch.utils.data.distributed.DistributedSampler(dataset_eval)
        dataloader_eval = torch.utils.data.DataLoader(
            dataset=dataset_eval,
            batch_size=model_zoo[source_model]['batch_size'],
            shuffle=False,
            num_workers=args.num_workers,
            sampler=sampler_eval,
            collate_fn=None,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        acc_np=np.zeros([55,2])
        clean_acc=validate(model, dataloader_eval, args)
        print('Top1 acc of clean images is: {0:>7.4f}'.format(clean_acc['top1']))
        acc_np[0,0]=112
        acc_np[0,1]=clean_acc['top1']

        i=1
        for radius in range(110, 2, -2):
            clean_acc=validate_fc(model, dataloader_eval, args, radius)
            acc_np[i, 0]=radius
            acc_np[i, 1]=clean_acc['top1']
            i+=1
        torch.distributed.barrier()
        if args.rank==0:
            np.save(os.path.join(args.output_dir, source_model+'.npy'), acc_np)
        torch.distributed.barrier()
   

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

def idea_lpf(input, radius):
    f_t=torch.fft.fft2(input)
    f_comp=torch.fft.fftshift(f_t)
    w,h=input.shape[-2], input.shape[-1]
    c_w, c_h=w//2, h//2
    filter=torch.zeros_like(input).cuda()
    for i in range(h):
        for j in range(w):
            if (i-c_h)**2+(j-c_w)**2<=radius**2:
                filter[:,:,i,j]=1
    f_comp_out=f_comp*filter
    output=torch.abs(torch.fft.ifft2(torch.fft.ifftshift(f_comp_out))).cuda()
    # output=torch.fft.irfft2(f_mag_out)
    return output

def validate_fc(model, loader, args, radius, log_suffix='clean acc'):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
                
        with torch.no_grad():
            if not radius==0:
                input=idea_lpf(input, radius)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust test script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
