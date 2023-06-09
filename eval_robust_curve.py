import os
import argparse
import numpy as np
import time
import torch
from collections import OrderedDict
import logging
import gdown

# torchvision func
from torchvision import transforms

# timm func
from timm.models import create_model
from timm.utils import AverageMeter, reduce_tensor, accuracy

from utils import random_seed, NormalizeByChannelMeanStd, distributed_init, create_logger, random_seed
from data.dataset import ImageNet
from model.resnet import resnet50, wide_resnet50_2
from model.resnet_denoise import get_FD
from model import vit_mae
from model.model_zoo import model_zoo
from adv.adv_utils import PgdAttackEps
from adv.autoattack_eps.autoattack import AutoAttackEps

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
#             model_kwargs['act_layer']=torch.nn.GELU
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


def load_attack(attack_name, init_kwargs):
    if attack_name=='pgd':
        attacker=PgdAttackEps(**init_kwargs)
    else:
        attacker=AutoAttackEps(**init_kwargs)

    return attacker

class DistortionBenchmark(object):
    ''' Distortion benchmark. '''

    def __init__(self, attack_name, model, batch_size, goal, distance_metric, iteration, distortion,
                 confidence=0.0, search_steps=5, binsearch_steps=10, _logger=None, **kwargs):
        ''' Initialize DistortionBenchmark.
        :param attack_name: The attack method's name. All valid values are ``'fgsm'``, ``'bim'``, ``'pgd'``, ``'mim'``,
            ``'cw'``, ``'deepfool'``, ``'nes'``, ``'spsa'``, ``'nattack'``.
        :param model: The classifier model to run the attack on.
        :param batch_size: Batch size for attack.
        :param goal: The adversarial goal for the attack method. All valid values are ``'t'`` for targeted attack,
            ``'tm'`` for targeted missclassification attack, and ``'ut'`` for untargeted attack.
        :param distance_metric: The adversarial distance metric for the attack method. All valid values are ``'l_2'``
            and ``'l_inf'``.
        :param session: The ``tf.Session`` instance for the attack to run in.
        :param distortion: Initial distortion. When doing search on attack magnitude, it is used as the starting point.
        :param confidence: For white box attacks, consider the adversarial as succeed only when the margin between top-2
            logits is larger than the confidence.
        :param search_steps: Search steps for finding an initial adversarial distortion.
        :param binsearch_steps: Binary search steps for refining the initial adversarial distortion.
        :param kwargs: Other keyword arguments to pass to the attack method's initialization function.
        '''
        self.attack_name = attack_name
        self.model=model

        self.batch_size, self.goal, self.distance_metric, self.iteration = batch_size, goal, distance_metric, iteration

        self.init_distortion = distortion
        self.confidence = confidence
        self.search_steps = search_steps
        self.binsearch_steps = binsearch_steps

        self._logger=_logger

        init_kwargs = dict()
        init_kwargs['model'] = self.model
        init_kwargs['batch_size'] = self.batch_size
        init_kwargs['goal'] = self.goal
        init_kwargs['distance_metric'] = self.distance_metric
        init_kwargs['iteration'] = self.iteration
        init_kwargs['_logger'] = self._logger
        for k, v in kwargs.items():
            init_kwargs[k] = v

        self.attack = load_attack(attack_name, init_kwargs)

        if self.attack_name in ('bim', 'pgd', 'mim'):
            self._run = self._run_binsearch_alpha
        elif self.attack_name in ('aa', 'deepfool'):
            self._run = self._run_binsearch_aa
        else:
            raise NotImplementedError

    def config(self, **kwargs):
        ''' (Re)config the attack method.
        :param kwargs: The key word arguments for the attack method's ``config()`` method.
        '''
        self.attack.config(**kwargs)

    def _run_binsearch_alpha(self, dataset, args):
        ''' The ``run`` method for bim, pgd and mim. '''
        # the attack is already configured in `config()`
        rs = []

        for i_batch, (xs, ys) in enumerate(dataset):
            xs=xs.cuda()
            ys=ys.cuda()

            xs_result = torch.zeros_like(xs).cuda()    # set xs_result to zeros initially, so that if the attack fails all the way down we could know it
            lo = torch.zeros(self.batch_size, dtype=torch.float).cuda()
            hi = lo + self.init_distortion            

            # use exponential search to find an adversarial magnitude
            for i in range(self.search_steps):
                # config the attack
                self.attack.config(magnitude=hi, alpha=hi / 4)
                # run the attack
                xs_adv = self.attack.batch_attack(xs, ys, None)
                logits = self.model(xs_adv)

                # check if attack succeed considering the confidence value
                if self.goal == 'ut' or self.goal == 'tm':
                    # for ut and tm goal, if the original label's logit is not the largest one, the example is
                    # adversarial.
                    # succ = logits.max(axis=1) - logits.take(ys_flatten) > self.confidence
                    ### check it
                    _, label=torch.max(logits, dim=1)
                    succ = label!=ys
                # update the advsearial examples
                xs_result[succ] = xs_adv[succ]
                # if failed, use a larger magnitude for next iteration
                not_succ = torch.logical_not(succ)
                lo[not_succ] = hi[not_succ]
                hi[not_succ] *= 2

                torch.distributed.barrier()
                succ_batch_tensor_list = [torch.zeros_like(succ) for _ in range(args.world_size)]
                torch.distributed.all_gather(succ_batch_tensor_list, succ)
                torch.distributed.barrier()
                succ_batch_all=torch.cat(succ_batch_tensor_list)

                if self._logger:
                    begin = i_batch * len(xs) * args.world_size
                    self._logger.info('search i_batch={}, n={}..{}: i={}, success_rate={:.3f}'.format(
                        i_batch, begin, begin + len(xs)*args.world_size - 1, i, torch.mean(succ_batch_all.float(), dtype=torch.float)))

                if succ_batch_all.all():
                    break

            torch.distributed.barrier()
            # run binsearch to find the minimal adversarial magnitude
            for i in range(self.binsearch_steps):
                # config the attack
                mi = (lo + hi) / 2
                self.attack.config(magnitude=mi, alpha=mi / 4)
                # run the attack
                xs_adv = self.attack.batch_attack(xs, ys, None)
                logits = self.model(xs_adv)
                # check if attack succeed considering the confidence value
                if self.goal == 'ut' or self.goal == 'tm':
                    # for ut and tm goal, if the original label's logit is not the largest one, the example is
                    # adversarial.
                    _, label=torch.max(logits, dim=1)
                    succ = label!=ys
                # update the advsearial examples
                xs_result[succ] = xs_adv[succ]
                # update hi (if succeed) or lo (if not)
                not_succ = torch.logical_not(succ)
                hi[succ] = mi[succ]
                lo[not_succ] = mi[not_succ]

                torch.distributed.barrier()
                succ_batch_tensor_list = [torch.zeros_like(succ) for _ in range(args.world_size)]
                torch.distributed.all_gather(succ_batch_tensor_list, succ)
                torch.distributed.barrier()
                succ_batch_all=torch.cat(succ_batch_tensor_list)

                if self._logger:
                    begin = i_batch * len(xs) * args.world_size
                    self._logger.info('binsearch i_batch={}, n={}..{}: i={}, success_rate={:.3f}'.format(
                        i_batch, begin, begin + len(xs)*args.world_size - 1, i, torch.mean(succ_batch_all.float(), dtype=torch.float)))
            torch.distributed.barrier()

            # to be more precise, make all the miscls samples to the original input
            with torch.no_grad():
                logits = self.model((xs-self.mean_tensor)/self.std_tensor)
                if self.goal == 'ut' or self.goal == 'tm':
                    _, label=torch.max(logits, dim=1)
                    mis_cls = label!=ys
                xs_result[mis_cls]=xs[mis_cls]
                lo[mis_cls]=hi[mis_cls]=torch.zeros_like(hi[mis_cls])

            # obtain the max eps
            rs_batch = []
            for x, x_result in zip(xs, xs_result):
                if (x_result == 0).all():  # all attacks failed
                    rs_batch.append(torch.tensor(100).float().cuda())
                else:
                    rs_batch.append(torch.max(torch.abs(x_result - x)))
            torch.distributed.barrier()
            rs_batch_tensor = torch.stack(rs_batch)
            rs_batch_tensor_list = [torch.zeros_like(rs_batch_tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(rs_batch_tensor_list, rs_batch_tensor)
            torch.distributed.barrier()
            rs_batch_all=torch.cat(rs_batch_tensor_list)

            # save the results of i_batch
            if args.rank==0:
                np.save(os.path.join(args.output_dir, 'result_{}.npy'.format(i_batch)), rs_batch_all.cpu().numpy())
            torch.distributed.barrier()
            rs.append(rs_batch_all)
        
        # combine all the batch tensors
        rs_tensor=torch.cat(rs)

        return rs_tensor.cpu().numpy()


    def _run_binsearch_aa(self, dataset, args):
        ''' The ``run`` method for autoattack '''
        rs = []

        for i_batch, (xs, ys) in enumerate(dataset):
            xs=xs.cuda()
            ys=ys.cuda()
            
            xs_result = torch.zeros_like(xs).cuda()
            lo = torch.zeros(self.batch_size, dtype=torch.float).cuda()
            hi = lo + self.init_distortion
            
            # use exponential search to find an adversarial magnitude
            for i in range(self.search_steps):
                # config the attack
                self.attack.config(magnitude=hi)

                # run the attack
                succ=torch.zeros(self.batch_size, dtype=torch.bool).cuda()
                for attack_to_run in self.attack.attacks_to_run:
                    self.attack.set_attack(attack_to_run)
                    xs_adv = self.attack.batch_attack(xs, ys, None)
                    logits = self.model(xs_adv)

                    # check if attack succeed considering the confidence value
                    if self.goal == 'ut' or self.goal == 'tm':
                        _, label=torch.max(logits, dim=1)
                        succ_temp = label!=ys
                    # update the advsearial examples and succ_any flag
                    xs_result[succ_temp] = xs_adv[succ_temp]
                    succ=torch.logical_or(succ, succ_temp)

                # if failed, use a larger magnitude for next iteration
                not_succ = torch.logical_not(succ)
                lo[not_succ] = hi[not_succ]
                hi[not_succ] *= 2

                torch.distributed.barrier()
                succ_batch_tensor_list = [torch.zeros_like(succ) for _ in range(args.world_size)]
                torch.distributed.all_gather(succ_batch_tensor_list, succ)
                torch.distributed.barrier()
                succ_batch_all=torch.cat(succ_batch_tensor_list)

                if self._logger:
                    begin = i_batch * len(xs) * args.world_size
                    self._logger.info('search i_batch={}, n={}..{}: i={}, success_rate={:.3f}'.format(
                        i_batch, begin, begin + len(xs)*args.world_size - 1, i, torch.mean(succ_batch_all.float(), dtype=torch.float)))
                
                if succ_batch_all.all():
                    break

            torch.distributed.barrier()
            # run binsearch to find the minimal adversarial magnitude
            for i in range(self.binsearch_steps):
                # config the attack
                mi = (lo + hi) / 2
                self.attack.config(magnitude=mi)

                # run the attack
                succ=torch.zeros(self.batch_size, dtype=torch.bool).cuda()
                for attack_to_run in self.attack.attacks_to_run:
                    self.attack.set_attack(attack_to_run)
                    xs_adv = self.attack.batch_attack(xs, ys, None)
                    logits = self.model(xs_adv)

                    # check if attack succeed considering the confidence value
                    if self.goal == 'ut' or self.goal == 'tm':
                        _, label=torch.max(logits, dim=1)
                        succ_temp = label!=ys
                    # update the advsearial examples and succ_any flag
                    xs_result[succ_temp] = xs_adv[succ_temp]
                    succ=torch.logical_or(succ, succ_temp)
                    # self._logger.info('position 2 attack {} allocate memory: {}'.format(attack_to_run, torch.cuda.memory_allocated()))

                # update hi (if succeed) or lo (if not)
                not_succ = torch.logical_not(succ)
                hi[succ] = mi[succ]
                lo[not_succ] = mi[not_succ]
                if self._logger:
                    begin = i_batch * len(xs) * args.world_size
                    self._logger.info('binsearch i_batch={}, n={}..{}: i={}, success_rate={:.3f}'.format(
                        i_batch, begin, begin + len(xs)*args.world_size - 1, i, torch.mean(succ.float(), dtype=torch.float)))
            torch.distributed.barrier()
            
            # to be more precise, make all the miscls samples to the original input
            with torch.no_grad():
                logits = self.model((xs-self.mean_tensor)/self.std_tensor)
                if self.goal == 'ut' or self.goal == 'tm':
                    _, label=torch.max(logits, dim=1)
                    mis_cls = label!=ys
                xs_result[mis_cls]=xs[mis_cls]
                lo[mis_cls]=hi[mis_cls]=torch.zeros_like(hi[mis_cls])


            # update max eps
            rs_batch=[]
            for sample_index in range(self.batch_size):
                x_result=xs_result[sample_index]
                if (x_result == 0).all():  # all attacks failed
                    rs_batch.append(torch.tensor(100).float().cuda())
                else:
                    rs_batch.append(hi[sample_index])
            torch.distributed.barrier()
            rs_batch_tensor = torch.stack(rs_batch)
            rs_batch_tensor_list = [torch.zeros_like(rs_batch_tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(rs_batch_tensor_list, rs_batch_tensor)
            torch.distributed.barrier()
            rs_batch_all=torch.cat(rs_batch_tensor_list)
            # save the results of i_batch
            if args.rank==0:
                np.save(os.path.join(args.output_dir, 'result_{}.npy'.format(i_batch)), rs_batch_all.cpu().numpy())
            torch.distributed.barrier()
            rs.append(rs_batch_all.cpu().numpy())

        # combine all the batch tensors
        rs_np=np.concatenate(rs)

        return rs_np


    def run(self, dataset, args):
        ''' Run the attack on the dataset.
        :param dataset: A ``tf.data.Dataset`` instance, whose first element is the unique identifier for the data point,
            second element is the image, third element is the ground truth label. If the goal is 'tm' or 't', a forth
            element should be provided as the target label for the attack.
        :param logger: A standard logger.
        :return: An numpy array of minimal distortion value for each input. If the attack method failed to generate
            adversarial example, the value is set to ``np.nan``.
        '''
        return self._run(dataset, args)


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
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL', help='Name of model to train')

    # data parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--crop-pct', default=0.875, type=float, metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', default=3, type=int, help='1: lanczos 2: bilinear 3: bicubic')

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', default=True)

    # evaluated datasets
    parser.add_argument('--imagenet_val_path', default='', type=str, help='path to imagenet validation dataset')
    parser.add_argument('--attack_names', type=str, nargs='*', default=[], help='autoattack, pgd')
    parser.add_argument('--iteration', default=100, type=int)
    parser.add_argument('--goal', default='ut', type=str)
    parser.add_argument('--distance_metric', default='l_inf', type=str)
    parser.add_argument('--distortion', default=0.00784, type=float)   #0.00784
    parser.add_argument('--confidence', default=0.0, type=float)
    parser.add_argument('--search_steps', default=5, type=int)
    parser.add_argument('--binsearch_steps', default=10, type=int)

    parser.add_argument('--output_dir', default='./test_out', type=str)

    return parser

def main(args):
    #distributed settings and logger
    if "WORLD_SIZE" in os.environ:
        args.world_size=int(os.environ["WORLD_SIZE"])
    args.distributed=args.world_size>1
    distributed_init(args)
    _logger = create_logger(args.output_dir, dist_rank=args.rank, name='main_benchmark', default_level=logging.INFO)
    
    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark = True

    t = []
    if args.input_size > 32:
        size = int(args.input_size/args.crop_pct)
        # size=args.input_size
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

    if args.imagenet_val_path:
        dataset_eval=ImageNet(root=args.imagenet_val_path, transform=test_transform)
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

        val_metrics = validate(model, dataloader_eval, args, log_suffix='ImageNet-Val', text=None, _logger=_logger)
        _logger.info(f"ImageNet-Val Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    
    for attack_name in args.attack_names:
        benchmark = DistortionBenchmark(attack_name, model, args.batch_size, args.goal, args.distance_metric, args.iteration, args.distortion, args.confidence, args.search_steps, args.binsearch_steps, _logger=_logger)
        dataset_eval=ImageNet(root=args.imagenet_val_path, transform=test_transform)
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
        result = benchmark.run(dataloader_eval, args)
        # save result
        if args.rank==0:
            np.save(os.path.join(args.output_dir, 'result.npy'), result)



def validate(model, loader, args, log_suffix='', mask=None, mapping=False, text=None, _logger=None):
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
        _logger.info(
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
