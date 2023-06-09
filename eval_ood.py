import os
import argparse
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets

# timm func
from timm.models import create_model
from timm.utils import AverageMeter, reduce_tensor, accuracy


from utils import random_seed, distributed_init, NormalizeByChannelMeanStd
from data.imagenet_a_utils import imagenet_a_mask
from data.imagenet_c_utils import get_ce_alexnet, data_loaders_names, get_mce_from_accuracy
from data.imagenet_r_utils import imagenet_r_mask, imagenet_r_wnids
from data.objectnet_utils import ObjectNetDataset, imageNetIDToObjectNetID
from data.imagenet_real_utils import ImageFolderReturnsPath, real_labels
from data.dataset import ImageNet
from model.resnet import resnet50, wide_resnet50_2
from model.resnet_denoise import get_FD
from model import vit_mae
from model.model_zoo import model_zoo

def get_model(model_name):
    backbone=model_zoo[model_name]['model']
    ckpt_dir=model_zoo[model_name]['ckpt']
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
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--crop-pct', default=0.875, type=float, metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', default=3, type=int, help='1: lanczos 2: bilinear 3: bicubic')
    
    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=6, type=int)

    # evaluated datasets
    parser.add_argument('--imagenet_val_path', default='', type=str, help='path to imagenet validation dataset')
    parser.add_argument('--imagenet_c_path', default='', type=str, help='path to imagenet curruption dataset')
    parser.add_argument('--imagenet_a_path', default='', type=str, help='path to imagenet natural adversarial dataset')
    parser.add_argument('--imagenet_200_path', default='', type=str, help='path to imagenet validation dataset')
    parser.add_argument('--imagenet_r_path', default='', type=str, help='path to imagenet rendition dataset')
    parser.add_argument('--imagenet_sketch_path', default='', type=str, help='path to imagenet sketch dataset')
    parser.add_argument('--imagenet_v2_path', default='', type=str, help='path to imagenetv2 dataset')
    parser.add_argument('--stylized_imagenet_path', default='', type=str, help='path to stylized imagenet dataset')
    parser.add_argument('--objectnet_path', default='', type=str, help='path to objectnet dataset')
    parser.add_argument('--imagenet_v', default='', type=str, help='path to imagenet validation dataset')
    parser.add_argument('--imagenet_v_meta', default='', type=str, help='path to imagenet validation dataset')
    
    return parser


def main(args):
    #distributed settings
    if "WORLD_SIZE" in os.environ:
        args.world_size=int(os.environ["WORLD_SIZE"])
    args.distributed=args.world_size>1
    distributed_init(args)

    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)
    torch.backends.cudnn.benchmark = True

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
    
    # ood evaluation
    if args.imagenet_val_path:
        # imagenet val (no fix label)
        dataset_val = datasets.ImageFolder(args.imagenet_val_path, transform=test_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
                
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_val, args, log_suffix='ImageNet-Val')
        print(f"ImageNet-Val Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

        # imagenet val (fix label)
        dataset_real = ImageFolderReturnsPath(args.imagenet_val_path, transform=test_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_real, shuffle=False)
                
        data_loader_real = torch.utils.data.DataLoader(
            dataset_real, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate_for_in_real(model, data_loader_real, args, log_suffix='ImageNet-Real')
        print(f"ImageNet-Real Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.objectnet_path:
        # objectnet transform
        objectnet_transform = transforms.Compose([transforms.Resize(args.input_size, interpolation=args.interpolation),
                            transforms.CenterCrop(args.input_size),
                            transforms.ToTensor()])

        dataset_objnet = ObjectNetDataset(args.objectnet_path, transform=objectnet_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_objnet, shuffle=False)
                
        data_loader_objnet = torch.utils.data.DataLoader(
            dataset_objnet, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_objnet, args, log_suffix='ObjectNet', mapping=True)
        print(f"ObjectNet Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.stylized_imagenet_path:
        if args.input_size == 224:
            stylized_in_transform = transforms.Compose([transforms.ToTensor()])
        else:
            stylized_in_transform = transforms.Compose([transforms.Resize(args.input_size, interpolation=args.interpolation),
                        transforms.CenterCrop(args.input_size),
                        transforms.ToTensor()])

        dataset_stylized_in = datasets.ImageFolder(args.stylized_imagenet_path, transform=stylized_in_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_stylized_in, shuffle=False)
        
        data_loader_stylized_in = torch.utils.data.DataLoader(
            dataset_stylized_in, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_stylized_in, args, log_suffix='Stylized-ImageNet')
        print(f"Stylized-ImageNet Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.imagenet_v2_path:
        dataset_val_v2 = datasets.ImageFolder(args.imagenet_v2_path, transform=test_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_val_v2, shuffle=False)
                
        data_loader_val_v2 = torch.utils.data.DataLoader(
            dataset_val_v2, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_val_v2, args, log_suffix='ImageNet-V2')
        print(f"ImageNet-V2 Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.imagenet_200_path:
        imagenet_val_location = args.imagenet_val_path
        imagenet_200_folder = args.imagenet_200_path
        create_symlinks_to_imagenet(imagenet_val_location, imagenet_200_folder)
        dataset_200 = datasets.ImageFolder(imagenet_200_folder, transform=test_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_200, shuffle=False)
                
        data_loader_200 = torch.utils.data.DataLoader(
            dataset_200, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_200, args, log_suffix='ImageNet-200', mask=imagenet_r_mask)
        print(f"ImageNet-200 Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.imagenet_r_path:
        dataset_inr = datasets.ImageFolder(args.imagenet_r_path, transform=test_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_inr, shuffle=False)
                
        data_loader_inr = torch.utils.data.DataLoader(
            dataset_inr, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_inr, args, log_suffix='ImageNet-R', mask=imagenet_r_mask)
        print(f"ImageNet-R Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")
    
    if args.imagenet_a_path:
        dataset_ina = datasets.ImageFolder(args.imagenet_a_path, transform=test_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_ina, shuffle=False)
                
        data_loader_ina = torch.utils.data.DataLoader(
            dataset_ina, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_ina, args, log_suffix='ImageNet-A', mask=imagenet_a_mask)
        print(f"ImageNet-A Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.imagenet_sketch_path:
        dataset_insk = datasets.ImageFolder(args.imagenet_sketch_path, transform=test_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_insk, shuffle=False)
                
        data_loader_insk = torch.utils.data.DataLoader(
            dataset_insk, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_insk, args, log_suffix='ImageNet-Sketch')
        print(f"ImageNet-Sketch Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.imagenet_c_path:
        result_dict = {}
        ce_alexnet = get_ce_alexnet()

        # transform for imagenet-c
        if args.input_size == 224:
            inc_transform = transforms.Compose([transforms.ToTensor()])
        else:
            inc_transform = transforms.Compose([transforms.Resize(args.input_size, interpolation=args.interpolation),
                        transforms.CenterCrop(args.input_size),
                        transforms.ToTensor()])

        for name, subdir in data_loaders_names.items():
            for severity in range(1, 6):
                inc_dataset = datasets.ImageFolder(os.path.join(args.imagenet_c_path, subdir, str(severity)), transform=inc_transform)
                sampler_val = None
                if args.distributed:
                    sampler_val = torch.utils.data.DistributedSampler(inc_dataset, shuffle=False)
                inc_data_loader = torch.utils.data.DataLoader(
                                inc_dataset, sampler=sampler_val,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=False
                            )
                test_stats = validate(model, inc_data_loader, args, log_suffix='ImageNet-C')
                print(f"Accuracy on the {name+'({})'.format(severity)}: {test_stats['top1']:.1f}%")
                result_dict[name+'({})'.format(severity)] = test_stats['top1']

        # calculate mCE
        mCE = 0
        counter = 0
        overall_acc = 0
        for name, _ in data_loaders_names.items():
            acc_top1 = 0
            for severity in range(1, 6):
                acc_top1 += result_dict[name+'({})'.format(severity)]
            acc_top1 /= 5
            CE = get_mce_from_accuracy(acc_top1, ce_alexnet[name])
            mCE += CE
            overall_acc += acc_top1
            counter += 1
            print("{0}: Top1 accuracy {1:.2f}, CE: {2:.2f}".format(
                    name, acc_top1, 100. * CE))
        
        overall_acc /= counter
        mCE /= counter
        print("Corruption Top1 accuracy {0:.2f}, mCE: {1:.2f}".format(overall_acc, mCE * 100.))

        # robust curve
        for severity in range(1, 6):
            acc_top1 = 0
            for name, _ in data_loaders_names.items():
                acc_top1 += result_dict[name+'({})'.format(severity)]
            acc_top1 /= len(data_loaders_names)
            print('Top1 accuracy of severity {0} is : {1:.2f}'.format(severity, acc_top1))
            
    if args.imagenet_v:
        # imagenet-v
        dataset_val = ImageNet(root=args.imagenet_v, meta_file=args.imagenet_v_meta, transform=test_transform)
        sampler_val = None
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
                
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_val, args, log_suffix='ImageNet-V')
        print(f"ImageNet-V Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")
    
def create_symlinks_to_imagenet(imagenet_val_location, imagenet_folder):
    if not os.path.exists(imagenet_folder):
        os.makedirs(imagenet_folder)
        folders_of_interest = imagenet_r_wnids  # os.listdir(folder_to_scan)
        for folder in folders_of_interest:
            os.symlink(imagenet_val_location + folder, imagenet_folder+folder, target_is_directory=True)
    else:
        print('Folder containing IID validation images already exists')


def validate(model, loader, args, log_suffix='', mask=None, mapping=False):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        input = input.cuda(args.device_id, non_blocking=True)
        target = target.cuda(args.device_id, non_blocking=True)
        with torch.no_grad():
            output = model(input)


        if mapping:
            _, prediction_class = output.topk(5, 1, True, True)
            prediction_class = prediction_class.data.cpu().tolist()
            for i in range(output.size(0)):
                imageNetIDToObjectNetID(prediction_class[i])

            prediction_class = torch.tensor(prediction_class).cuda(args.device_id, non_blocking=True)
            prediction_class = prediction_class.t()
            correct = prediction_class.eq(target.reshape(1, -1).expand_as(prediction_class))
            acc1, acc5 = [correct[:k].reshape(-1).float().sum(0) * 100. / output.size(0) for k in (1, 5)]

        else:
            if mask is None:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            else:
                acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))

        if args.distributed:
            acc1 = reduce_tensor(acc1, args.world_size)
            acc5 = reduce_tensor(acc5, args.world_size)

        torch.cuda.synchronize()

        top1_m.update(acc1.item(), input.size(0))
        top5_m.update(acc5.item(), input.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

        log_name = 'Test ' + log_suffix
        print(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
            'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

def validate_for_in_real(model, loader, args, log_suffix=''):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, img_paths) in enumerate(loader):

        input = input.cuda(args.device_id, non_blocking=True)
        with torch.no_grad():
            output = model(input)

        is_correct = {k: [] for k in (1, 5)}

        _, pred_batch = output.topk(5, 1, True, True)

        pred_batch = pred_batch.cpu().numpy()
        sample_idx = 0
        for pred in pred_batch:
            filename = os.path.basename(img_paths[sample_idx])
            if real_labels[filename]:
                for k in (1, 5):
                    is_correct[k].append(
                        any([p in real_labels[filename] for p in pred[:k]]))
            sample_idx += 1

        acc1 = torch.tensor(float(np.mean(is_correct[1])) * 100.).cuda(args.device_id, non_blocking=True)
        acc5 = torch.tensor(float(np.mean(is_correct[5])) * 100.).cuda(args.device_id, non_blocking=True)

        if args.distributed:
            acc1 = reduce_tensor(acc1, args.world_size)
            acc5 = reduce_tensor(acc5, args.world_size)

        torch.cuda.synchronize()

        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

        log_name = 'Test ' + log_suffix
        print(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
            'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust test script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)