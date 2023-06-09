import math
import time

import numpy as np
import torch

# from .other_utils import Logger
# from third_party.autoattack.autoattack import checks

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

funcs = {'grad': 0,
    'backward': 0,
    #'enable_grad': 0
    '_make_grads': 0,
    }

checks_doc_path = 'flags_doc.md'


def check_randomized(model, x, y, bs=250, n=5, alpha=1e-4, logger=None):
    acc = []
    corrcl = []
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            output = model(x)
            corrcl_curr = (output.max(1)[1] == y).sum().item()
            corrcl.append(corrcl_curr)
            outputs.append(output / (L2_norm(output, keepdim=True) + 1e-10))
    acc = [c != corrcl_curr for c in corrcl]
    max_diff = 0.
    for c in range(n - 1):
        for e in range(c + 1, n):
            diff = L2_norm(outputs[c] - outputs[e])
            max_diff = max(max_diff, diff.max().item())
            #print(diff.max().item(), max_diff)
    if any(acc) or max_diff > alpha:
        msg = 'it seems to be a randomized defense! Please use version="rand".' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_range_output(model, x, alpha=1e-5, logger=None):
    with torch.no_grad():
        output = model(x)
    fl = [output.max() < 1. + alpha, output.min() >  -alpha,
        ((output.sum(-1) - 1.).abs() < alpha).all()]
    if all(fl):
        msg = 'it seems that the output is a probability distribution,' +\
            ' please be sure that the logits are used!' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    return output.shape[-1]


def check_zero_gradients(grad, logger=None):
    z = grad.view(grad.shape[0], -1).abs().sum(-1)
    #print(grad[0, :10])
    if (z == 0).any():
        msg = f'there are {(z == 0).sum()} points with zero gradient!' + \
            ' This might lead to unreliable evaluation with gradient-based attacks.' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_square_sr(acc_dict, alpha=.002, logger=None):
    if 'square' in acc_dict.keys() and len(acc_dict) > 2:
        acc = min([v for k, v in acc_dict.items() if k != 'square'])
        if acc_dict['square'] < acc - alpha:
            msg = 'Square Attack has decreased the robust accuracy of' + \
                f' {acc - acc_dict["square"]:.2%}.' + \
                ' This might indicate that the robustness evaluation using' +\
                ' AutoAttack is unreliable. Consider running Square' +\
                ' Attack with more iterations and restarts or an adaptive attack.' + \
                f' See {checks_doc_path} for details.'
            if logger is None:
                warnings.warn(Warning(msg))
            else:
                logger.log(f'Warning: {msg}')


''' from https://stackoverflow.com/questions/26119521/counting-function-calls-python '''
def tracefunc(frame, event, args):
    if event == 'call' and frame.f_code.co_name in funcs.keys():
        funcs[frame.f_code.co_name] += 1

        
def check_dynamic(model, x, is_tf_model=False, logger=None):
    if is_tf_model:
        msg = 'the check for dynamic defenses is not currently supported'
    else:
        msg = None
    sys.settrace(tracefunc)
    model(x)
    sys.settrace(None)
    #for k, v in funcs.items():
    #    print(k, v)
    if any([c > 0 for c in funcs.values()]):
        msg = 'it seems to be a dynamic defense! The evaluation' + \
            ' with AutoAttack might be insufficient.' + \
            f' See {checks_doc_path} for details.'
    if not msg is None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    #sys.settrace(None)


def check_n_classes(n_cls, attacks_to_run, apgd_targets, fab_targets,
    logger=None):
    msg = None
    if 'apgd-dlr' in attacks_to_run or 'apgd-t' in attacks_to_run:
        if n_cls <= 2:
            msg = f'with only {n_cls} classes it is not possible to use the DLR loss!'
        elif n_cls == 3:
            msg = f'with only {n_cls} classes it is not possible to use the targeted DLR loss!'
        elif 'apgd-t' in attacks_to_run and \
            apgd_targets + 1 > n_cls:
            msg = f'it seems that more target classes ({apgd_targets})' + \
                f' than possible ({n_cls - 1}) are used in {"apgd-t".upper()}!'
    if 'fab-t' in attacks_to_run and fab_targets + 1 > n_cls:
        if msg is None:
            msg = f'it seems that more target classes ({apgd_targets})' + \
                f' than possible ({n_cls - 1}) are used in FAB-T!'
        else:
            msg += f' Also, it seems that too many target classes ({apgd_targets})' + \
                f' are used in {"fab-t".upper()} ({n_cls - 1} possible)!'
    if not msg is None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()

class AutoAttack():
    def __init__(self, model,steps,query, norm='Linf', eps=.3, seed=None, verbose=False,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 device='cuda', log_path=None):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.steps = steps
        self.query = query
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)

        if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
        
        if not self.is_tf_model:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=self.steps, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                device=self.device, logger=self.logger)
            
            from .fab_pt import FABAttack_PT
            self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=self.steps, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=False, device=self.device)
        
            from .square import SquareAttack
            self.square = SquareAttack(self.model, p_init=.8, n_queries=self.query, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=self.steps, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                logger=self.logger)
    
        else:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=self.steps, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)
            
            from .fab_tf import FABAttack_TF
            self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=self.steps, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=False, device=self.device)
        
            from .square import SquareAttack
            self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=self.query, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=self.steps, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)
    
        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)
        
    def get_logits(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed
    
    def run_standard_evaluation(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        # checks on type of defense
        if self.version != 'rand':
            checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device),
                y_orig[:bs].to(self.device), bs=bs, logger=self.logger)
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device),
            logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), self.is_tf_model,
            logger=self.logger)
        checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
            self.fab.n_target_classes, logger=self.logger)
        
        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            y_adv = torch.empty_like(y_orig)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                output = self.get_logits(x).max(dim=1)[1]
                y_adv[start_idx: end_idx] = output
                correct_batch = y.eq(output)
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
            robust_accuracy_dict = {'clean': robust_accuracy}
            
            if self.verbose:
                self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
                    
            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    
                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)
                    
                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True
                    
                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    else:
                        raise ValueError('Attack not supported')
                
                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)    
                        self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
                
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))
                    
            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)
            
            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv
        
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()
            
        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))
        
        return acc.item() / x_orig.shape[0]
        
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        
        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            if return_labels:
                adv[c] = (x_adv, y_adv)
            else:
                adv[c] = x_adv
            if verbose_indiv:    
                acc_indiv  = self.clean_accuracy(x_adv, y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))
        
        return adv
        
    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))
        
        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            #self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = self.query
        
        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = self.query
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))
        
        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20

