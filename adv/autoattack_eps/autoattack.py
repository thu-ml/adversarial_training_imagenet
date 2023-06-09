from .autopgd_base import APGDAttack, APGDAttack_targeted
from .fab_pt import FABAttack_PT
from .square import SquareAttack
import time

class AutoAttackEps(object):
    def __init__(self, model, batch_size, goal, distance_metric, magnitude=4/255, alpha=1/255, iteration=100, seed=None, attacks_to_run=[], version='standard', device=None, _logger=None):
        self.model = model
        self.batch_size=batch_size
        self.goal=goal
        self.distance_metric = distance_metric
        assert self.distance_metric in ['l_inf', 'l_1', 'l_2']
        self.magnitude=magnitude
        self.alpha=alpha
        self.iteration=iteration
        self.seed = seed
        self.attacks_to_run = attacks_to_run
        self.attack_to_run=None
        self.version = version
        dict_metric={'l_inf':'Linf','l_1':'L1','l_2':'L2'}
        self.distance_metric=dict_metric[self.distance_metric]
        self.device=device
        self._logger=_logger

        if self.version in ['standard', 'plus', 'rand'] and self.attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
        
        self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
            eps=self.magnitude, norm=self.distance_metric, eot_iter=1, rho=.75, seed=self.seed, logger=self._logger)

        self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.magnitude, seed=self.seed,
            norm=self.distance_metric, verbose=False, logger=self._logger)
    
        self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.magnitude, norm=self.distance_metric,
            n_restarts=1, seed=self.seed, verbose=False, resc_schedule=False, logger=self._logger)
            
        self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            eps=self.magnitude, norm=self.distance_metric, eot_iter=1, rho=.75, seed=self.seed, logger=self._logger)
    
        if self.version in ['standard', 'plus', 'rand']:
            self.set_version(version)

    def config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # modify the magnitude
        self.apgd.eps=self.magnitude
        self.fab.eps=self.magnitude
        # self.square.eps=self.magnitude
        self.apgd_targeted.eps=self.magnitude
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed 

    def set_attack(self, attack_to_run):
        self.attack_to_run=attack_to_run

    def batch_attack(self, x_orig, y_orig, target_label):
        prev_training = bool(self.model.training)
        self.model.eval()
                     
        x=x_orig.cuda()
        y=y_orig.cuda()

        # run attack
        if self.attack_to_run == 'apgd-ce':
            # apgd on cross-entropy loss
            self.apgd.loss = 'ce'
            self.apgd.seed = self.get_seed()
            adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
        elif self.attack_to_run == 'apgd-dlr':
            # apgd on dlr loss
            self.apgd.loss = 'dlr'
            self.apgd.seed = self.get_seed()
            adv_curr = self.apgd.perturb(x, y) #cheap=True
            
        elif self.attack_to_run == 'fab':
            # fab
            self.fab.targeted = False
            self.fab.seed = self.get_seed()
            adv_curr = self.fab.perturb(x, y)
            
        elif self.attack_to_run == 'square':
            # square
            self.square.seed = self.get_seed()
            adv_curr = self.square.perturb(x, y)
            
        elif self.attack_to_run == 'apgd-t':
            # targeted apgd
            self.apgd_targeted.seed = self.get_seed()
            adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True
            
        elif self.attack_to_run == 'fab-t':
            # fab targeted
            self.fab.targeted = True
            self.fab.n_restarts = 1
            self.fab.seed = self.get_seed()
            adv_curr = self.fab.perturb(x, y)
            
        else:
            raise ValueError('Attack not supported')

        self._logger.info('{} attack has finished.'.format(self.attack_to_run))

        if prev_training:
            self.model.train()

        
        return adv_curr
        
    def set_version(self, version='standard'):
        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.distance_metric in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.distance_metric in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.square.n_queries = 5000
        
        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.distance_metric in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.distance_metric))
        
        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20