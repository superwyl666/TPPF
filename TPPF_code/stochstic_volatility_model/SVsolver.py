#pylint: disable=invalid-name, no-member, too-many-arguments, missing-docstring
#pylint: disable=too-many-instance-attributes, not-callable, no-else-return
#pylint: disable=inconsistent-return-statements, too-many-locals, too-many-return-statements
#pylint: disable=too-many-statements, too-many-public-methods

from copy import deepcopy
from datetime import date
import json
import numpy as np
import os
import time
import torch as pt
import math

from SVfunction_space import DenseNet, Linear, NN, NN_Nik, SingleParam, MySequential, DenseNet_tanh, DenseNet1, DenseNet2, DenseNet3
from SVutilities import compute_test_error, do_importance_sampling, do_importance_sampling_me


class mySolver_problemdependent():

    def __init__(self, name, problem, lr=0.01, L=500, K=500, delta_t=0.25,
                 approx_method='control', loss_method='relative-entropy', time_approx='outer',
                 learn_Y_0=False, adaptive_forward_process=True, detach_forward=False,
                 early_stopping_time=10000, random_X_0=False, compute_gradient_variance=0,
                 IS_variance_K=0, IS_variance_iter=1, metastability_logs=None, print_every=100, plot_trajectories=None,
                 seed=42, save_results=False, u_l2_error_flag=True, log_gradient=False,  verbose=False, T = 0.5,
                 sampling_method = 'untwisted', train_goal = 'RE', d = 2, OBsigma2 = 1, y_observed = [],
                 K_BPF_large = 10000, K_BPF_small = 200, phi0_BPF = 1, replicate_num = 1000, Z_true = 1, resample_kappa = 1):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = d
        self.T = T
        # self.X_0 = problem.X_0
        self.X_0 = pt.ones(self.d)
        self.Y_0 = pt.tensor([0.0])
        self.X_u_opt = None
        

        self.OBsigma2 = OBsigma2
        self.y_observed = y_observed

        self.Z_true = Z_true

        self.resample_kappa = resample_kappa

        self.average_ESSnum = 0
        

        self.sampling_method = sampling_method
        self.train_goal = train_goal

        # hyperparameters
        self.device = pt.device('cuda')
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(self.device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(self.device)
        self.N = int(np.floor(self.T / self.delta_t_np)) # number of steps
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size
        self.random_X_0 = random_X_0

        self.K_BPF_large = K_BPF_large#particle number for BPF
        self.K_BPF_small = K_BPF_small
        self.phi0_BPFL = phi0_BPF
        self.replicate_num = replicate_num


        # learning properties
        self.loss_method = loss_method
        self.approx_method = approx_method
        self.learn_Y_0 = learn_Y_0
        self.adaptive_forward_process = adaptive_forward_process
        self.detach_forward = detach_forward
        self.early_stopping_time = early_stopping_time
        

        self.has_ref_solution = hasattr(problem, 'u_true')
        self.u_l2_error_flag = u_l2_error_flag
        if self.has_ref_solution is False:
            self.u_l2_error_flag = False

        if self.loss_method == 'relative_entropy':
            self.adaptive_forward_process = True
        if self.loss_method == 'cross_entropy':
            self.learn_Y_0 = False

        # printing and logging
        self.print_every = print_every
        self.verbose = verbose
        self.verbose_NN = False
        self.save_results = save_results
        self.compute_gradient_variance = compute_gradient_variance
        self.IS_variance_K = IS_variance_K
        self.IS_variance_iter = IS_variance_iter
        self.metastability_logs = metastability_logs
        self.plot_trajectories = plot_trajectories
        self.metastability_logs = metastability_logs
        self.log_gradient = log_gradient
        self.print_gradient_norm = False

        # function approximation
        self.Phis = []
        self.time_approx = time_approx

        pt.manual_seed(seed)
        if self.approx_method == 'control':
            '''
            define NN here: at each time i z_n[i] = NN[i] from R^d to R^d
            or NN = NN(t,x) 
            output of NN is changed into 1d
            '''
            if self.time_approx == 'outer':
                self.z_n = [DenseNet(d_in=self.d, d_out=self.d, lr=self.lr, seed=seed) for i in range(self.N)]
                #self.z_n = [DenseNet_tanh(d_in=self.d, d_out=1, lr=self.lr, seed=seed) for i in range(self.N)]
            elif self.time_approx == 'inner':
                #self.co_z_n = DenseNet1(d_in= 1, d_out=1, lr=self.lr, seed=123)#coefficient:ck
                self.mean_z_n = DenseNet2(d_in= 1, d_out=self.d, lr=self.lr, seed=self.seed)#mean
                self.var_z_n = DenseNet3(d_in= 1, d_out=1, lr=self.lr, seed=self.seed)#var
                '''
                (consider the simple case each d is independent,i.e. b(X) = a Id cdot X)
                '''
                #self.z_n = MySequential(d_in=self.d + 1, d_out=self.d, lr=self.lr, seed=123)
                #self.z_n = MySequential(d_in=self.d + 1, d_out=1, lr=self.lr, seed=123)


        self.update_Phis()

        for phi in self.Phis:
            phi.train()

        if self.verbose_NN is True:
            if self.time_approx == 'outer':
                print('%d NNs, %d parameters in each network, total parameters = %d'
                      % (self.N, self.p, self.p * self.N))
            else:
                print('%d NNs, %d parameters in each network, total parameters = %d'
                      % (1, self.p, self.p))

        # logging
        self.Y_0_log = []

        self.loss_log = []
        self.loss_log_CE = []
        self.loss_log_RE_untwisted = []
        self.loss_log_CERE = []
        self.relvar_log = []
        self.loss_log_REdirect = []

        self.TPF_Z = []
        self.TPF_logZ = []


        self.u_L2_loss = []
        self.IS_rel_log = []
        self.times = []
        self.grads_rel_error_log = []
        self.particles_close_to_target = []

    def b(self, x):
        # bi, bj = pt.meshgrid(pt.arange(self.d), pt.arange(self.d))
        # alpha = 0.05
        # bA = alpha ** (pt.abs(bi - bj) + 1)
        # bA = bA.to(self.device)

        # bbeta = 0.1
        # bA = pt.eye(self.d).to(self.device) 
        # bA[-1, -1] = bbeta

        #return -pt.mm(bA, x.t()).t()
        #return (pt.mm(bA, x.t()).t() - x) / self.delta_t
        # return self.problem.b(x)
        return -x
        # AAindices = pt.arange(self.d)
        # AAi = AAindices.view(-1, 1)    # Column vector
        # AAj = AAindices.view(1, -1)    # Row vector
        # AA = 0.5 ** (pt.abs(AAi - AAj) + 1).to(self.device)
        # return pt.mm(AA, x.t()).t()

    def sigma(self, x):
        return self.problem.sigma(x)

    def h(self, t, x, y, z):
        return self.problem.h(t, x, y, z)

    def f(self, xx, t):
        #return self.problem.f(xx, t)#=pt.sum(x.t() * pt.mm(self.P, x.t()), 0)
        '''
        ft = -log gt
        gt = N(cdot,yt,sigma) 
        so ft = |x-yt|^2 / (2*sigma2)
        here the input is x-yt
        '''
        return pt.sum(xx.t() * pt.mm(pt.eye(self.d).to(self.device), xx.t()), 0) / (2 * self.OBsigma2)

    def g(self, x):
        return self.problem.g(x)

    def u_true(self, x, t):
        return self.problem.u_true(x, t)

    def v_true(self, x, t):
        return self.problem.v_true(x, t)

    def update_Phis(self):
        if self.approx_method == 'control':
            if self.learn_Y_0 is True:
                if self.time_approx == 'outer':
                    self.Phis = self.z_n + [self.y_0]
                elif self.time_approx == 'inner':
                    self.Phis = [self.z_n, self.y_0]
            else:
                '''
                outer means t in outside NN
                inner means NN = NN(t,x)
                '''
                if self.time_approx == 'outer':
                    self.Phis = self.z_n
                elif self.time_approx == 'inner':
                    #self.Phis = [self.z_n]
                    self.Phis = [self.mean_z_n, self.var_z_n]
        for phi in self.Phis:
            phi.to(self.device)
        '''
        number of the parameters 
        '''
        self.p = sum([np.prod(params.size()) for params in
          filter(lambda params: params.requires_grad,
                 self.Phis[0].parameters())])
        print(self.p, 'parameters in total.')
        if self.log_gradient is True:
            self.gradient_log = pt.zeros(self.L, self.p)

    def loss_function(self, X, Y, Z_sum, l):
        if self.loss_method == 'relative_entropy':
            #return (Z_sum + self.g(X)).mean() #changed into particle filter's setting, g is moving inside Z_sum
            return (Z_sum ).mean()
      

    def zero_grad(self):
        for phi in self.Phis:
            phi.optim.zero_grad()

    def optimization_step(self):
        for phi in self.Phis:
            phi.optim.step()

    def gradient_descent(self, X, Y, Z_sum, l, additional_loss):
        self.zero_grad()

        loss = self.loss_function(X, Y, Z_sum, l) + additional_loss
        loss.backward()
        self.optimization_step()
        return loss

    def flatten_gradient(self, k, grads, grads_flat):
        i = 0
        for grad in grads:
            grad_flat = grad.reshape(-1)
            j = len(grad_flat)
            grads_flat[k, i:i + j] = grad_flat
            i += j
        return grads_flat

    def get_gradient_variances(self, X, Y):
        grads_mean = pt.zeros(self.N, self.p)
        grads_var = pt.zeros(self.N, self.p)

        for n in range(self.N):

            grads_Y_flat = pt.zeros(self.K, self.p)

            for k in range(self.K):
                self.zero_grad()
                Y[k].backward(retain_graph=True)

                grad_Y = [params.grad for params in list(filter(lambda params:
                                                                params.requires_grad,
                                                                self.z_n[n].parameters()))
                          if params.grad is not None]

                grads_Y_flat = self.flatten_gradient(k, grad_Y, grads_Y_flat)

            grads_g_X_flat = pt.zeros(self.K, self.p)

            if self.adaptive_forward_process is True:

                for k in range(self.K):
                    self.zero_grad()
                    self.g(X[0, :].unsqueeze(0)).backward(retain_graph=True)

                    grad_g_X = [params.grad for params in list(filter(lambda params:
                                                                      params.requires_grad,
                                                                      self.z_n[n].parameters()))
                                if params.grad is not None]

                    grads_g_X_flat = self.flatten_gradient(k, grad_g_X, grads_g_X_flat)

            if self.loss_method == 'moment':
                grads_flat = 2 * (Y - self.g(X)).unsqueeze(1) * (grads_Y_flat - grads_g_X_flat)
            elif self.loss_method == 'log-variance':
                grads_flat = 2 * (((Y - self.g(X)).unsqueeze(1)
                                   - pt.mean((Y - self.g(X)).unsqueeze(1), 0).unsqueeze(0))
                                  * (grads_Y_flat - grads_g_X_flat
                                     - pt.mean(grads_Y_flat - grads_g_X_flat, 0).unsqueeze(0)))

            grads_mean[n, :] = pt.mean(grads_flat, dim=0)
            grads_var[n, :] = pt.var(grads_flat, dim=0)

        grads_rel_error = pt.sqrt(grads_var) / grads_mean
        grads_rel_error[grads_rel_error != grads_rel_error] = 0
        return grads_rel_error

    def state_dict_to_list(self, sd):
        sd_list = {}
        for name in sd:
            if type(sd[name]) == pt.Tensor:
                sd_list[name] = sd[name].detach().cpu().numpy().tolist()
            else:
                sd_list[name] = sd[name]
        return sd_list

    def list_to_state_dict(self, l):
         return {param: pt.tensor(l[param]) if type(l[param]) == list else l[param] for param in l}

    def save_logs(self, model_name='model'):
        # currently does not work for all modi
        logs = {'name': self.name, 'date': self.date, 'd': self.d, 'T': self.T,
                'seed': self.seed, 'delta_t': self.delta_t_np, 'N': self.N, 'lr': self.lr,
                'K': self.K, 'loss_method': self.loss_method, 'learn_Y_0': self.learn_Y_0,
                'adaptive_forward_process': self.adaptive_forward_process,
                'Y_0_log': self.Y_0_log, 'loss_log': self.loss_log, 'u_L2_loss': self.u_L2_loss,
                'Phis_state_dict': [self.state_dict_to_list(z.cpu().state_dict()) for z in self.Phis]}

        path_name = 'logs/%s_%s_%s.json' % (model_name, self.name, self.date)
        i = 1
        while os.path.isfile(path_name):
            i += 1
            path_name = 'logs/%s_%s_%s_%d.json' % (model_name, self.name, self.date, i)

        with open(path_name, 'w') as f:
            json.dump(logs, f, indent=2)

    def save_networks(self):
        data_dict = {}
        idx = 0
        for z in self.Phis:
            key = 'nn%d' % idx
            data_dict[key] = z.state_dict()
            idx += 1
        path_name = 'output/%s_%s.pt' % (self.name, self.date)
        pt.save(data_dict, path_name)
        print('\nnetworks data has been stored to file: %s' % path_name)

    def load_networks(self, cp_name):
        print('\nload network data from file: %s' % cp_name)
        checkpoint = pt.load(cp_name)
        idx = 0
        for z in self.Phis:
            key = 'nn%d' % idx
            z.load_state_dict(checkpoint[key])
            z.eval()
            idx += 1

    def compute_grad_Y(self, X, n):
        Y_n_eval = self.Y_n(X, n).squeeze(1).sum()
        Y_n_eval.backward(retain_graph=True)
        Z, = pt.autograd.grad(Y_n_eval, X, create_graph=True)
        Z = pt.mm(self.sigma(X), Z.t()).t()
        return Z

    def Y_n(self, X, t):
        n = int(np.ceil(t / self.delta_t))
        if self.time_approx == 'outer':
            return self.y_n[n](X)
        elif self.time_approx == 'inner':
            t_X = pt.cat([pt.ones([X.shape[0], 1]) * t, X], 1)
            return self.y_n[0](t_X)

    def Z_n_(self, X, n):
        if self.approx_method == 'control':
            if self.time_approx == 'outer':
                n = max(0, min(n, self.N - 1))
                return self.z_n[n](X)
            elif self.time_approx == 'inner':
                t_X = pt.cat([pt.ones([X.shape[0], 1]).to(self.device) * n * self.delta_t, X], 1)
                return self.z_n(t_X)


    

    def phi_Z_n_(self, X, n):
        return 1 * pt.pow(2 * pt.pi * self.var_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device)), - self.d / 2)\
              * pt.exp(-pt.sum((X - self.mean_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device)).expand(X.shape[0], -1))**2,dim=1, keepdim=True)\
                        / (2 * self.var_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device))))
    

    def tildephi_Z_n_(self, X, n):
        return 1 * pt.pow(2 * pt.pi * (self.delta_t + self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device))), - self.d / 2)\
              * pt.exp(-pt.sum((X + self.b(X) * self.delta_t - self.mean_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)).expand(X.shape[0], -1))**2,dim=1, keepdim=True)\
                        / (2 * (self.delta_t + self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)))))


    def log_phi_Z_n_(self, X, n):
        return (- self.d / 2) * pt.log(2 * pt.pi * self.var_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device)))\
              + (-pt.sum((X - self.mean_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device)).expand(X.shape[0], -1))**2,dim=1, keepdim=True)\
                        / (2 * self.var_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device))))
    

    def log_tildephi_Z_n_(self, X, n):
        return (- self.d / 2) * pt.log(2 * pt.pi * (self.delta_t + self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device))))\
              + (-pt.sum((X + self.b(X) * self.delta_t - self.mean_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)).expand(X.shape[0], -1))**2,dim=1, keepdim=True)\
                        / (2 * (self.delta_t + self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)))))



    def phi_Z_n_scale(self, X, n):
        return  pt.exp(-pt.sum((X - self.mean_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device)).expand(X.shape[0], -1))**2,dim=1, keepdim=True)\
                        / (2 * self.var_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device))))
    

    def tildephi_Z_n_scale(self, X, n):
        return 1 * pt.pow(2 * pt.pi * (self.delta_t + self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)))\
                           / self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)), - self.d / 2)\
              * pt.exp(-pt.sum((X + self.b(X) * self.delta_t - self.mean_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)).expand(X.shape[0], -1))**2,dim=1, keepdim=True)\
                        / (2 * (self.delta_t + self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)))))


    def log_phi_Z_n_scale(self, X, n):
        return -pt.sum((X - self.mean_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device)).expand(X.shape[0], -1))**2,dim=1, keepdim=True)\
                        / (2 * self.var_z_n(pt.tensor([[n]], dtype=pt.float32).to(self.device)))
    

    def log_tildephi_Z_n_scale(self, X, n):
        return - self.d / 2 * pt.log(2 * pt.pi * (self.delta_t + self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)))\
                                      / self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)))\
              + (-pt.sum((X + self.b(X) * self.delta_t - self.mean_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)).expand(X.shape[0], -1))**2,dim=1, keepdim=True)\
                        / (2 * (self.delta_t + self.var_z_n(pt.tensor([[n+1]], dtype=pt.float32).to(self.device)))))




    def Z_n(self, X, t):
        n = int(pt.ceil(t / self.delta_t))
        return self.Z_n_(X, n)

    '''
    initialize
    '''
    def initialize_training_data(self):
        X = self.X_0.repeat(self.K, 1).to(self.device)
        if self.random_X_0 is True:
            X = pt.randn(self.K, self.d).to(self.device)
        Y = self.Y_0.repeat(self.K).to(self.device)
        if self.approx_method == 'value_function':
            X = pt.autograd.Variable(X, requires_grad=True)
            Y = self.Y_n(X, 0)[:, 0]
        elif self.learn_Y_0 is True:
            Y = self.y_0(X)
            self.Y_0_log.append(Y[0].item())
        Z_sum = pt.zeros(self.K).to(self.device)
        u_L2 = pt.zeros(self.K).to(self.device)
        u_int = pt.zeros(self.K).to(self.device)
        u_W_int = pt.zeros(self.K).to(self.device)
        double_int = pt.zeros(self.K).to(self.device)

        xi = pt.randn(self.K, self.d, self.N + 1).to(self.device)
        return X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi

    def train_LSE_with_reference(self):
        if self.approx_method != 'control':
            print('only learn control with reference solution!')
        if self.has_ref_solution == False:
            print('reference solution is needed!')

        print('\nd = %d, L = %d, K = %d, delta_t = %.2e, N = %d, lr = %.2e, %s, %s, %s, %s\n' % (self.d, self.L, self.K, self.delta_t_np, self.N, self.lr, self.approx_method, self.time_approx, self.loss_method, 'adaptive' if self.adaptive_forward_process else ''))

        xb = 2.0
        X = pt.linspace(-xb, xb, 200).unsqueeze(1)

        for l in range(self.L):
            t_0 = time.time()
            loss = 0.0
            for n in range(self.N):
                loss += pt.sum((- self.Z_n_(X, n) - pt.tensor(self.u_true(X, n * self.delta_t_np)).float())**2) * self.delta_t
            self.zero_grad()
            loss.backward()
            self.optimization_step()

            self.loss_log.append(loss.item())

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                string = ('%d - loss: %.3e - time/iter: %.2fs' % (l, self.loss_log[-1],
                          np.mean(self.times[-self.print_every:])))
                print(string + ' - gradient l_inf: %.3e' %
                      (np.array([max([pt.norm(params.grad.data, float('inf')).item() for params
                                      in filter(lambda params: params.requires_grad,
                                                phi.parameters())])
                                 for phi in self.Phis]).max()))

        #   self.save_networks()

    def train(self):
        print('lr=',self.lr)

        pt.manual_seed(self.seed)


        '''
        first, run a BPF with large amount of particles to obtain am acurate Z
        '''
       
        X_BPF = self.X_0.repeat(self.K_BPF_large, 1).to(self.device)

        y_observed_KBPFL = []#define y_OB for K_BPF_large particles
        for n in range(self.N + 1):
            y_observed_KBPFL.append(self.y_observed[n].repeat(self.K_BPF_large,1).to(self.device))

        # print(X_BPF.size(),y_observed_KBPFL[0].size())
        # print(asdfasdf)
        g_prod = 1
        #g_prod = pt.ones(self.K_BPF_large).to(self.device)
        for n in range(self.N+1):
            g_k = pt.exp(-self.f(X_BPF - y_observed_KBPFL[n],n * self.delta_t))
            g_prod *= g_k.mean().item()
            #g_prod *= g_k
            
            
            #resample
            weights = g_k / pt.sum(g_k)
            indices = pt.multinomial(weights, self.K_BPF_large, replacement=True)
            X_BPF = X_BPF[indices]
            
            #print(indices)

            #update particle position
            X_BPF = X_BPF + self.delta_t * self.b(X_BPF) + self.sq_delta_t * pt.randn_like(X_BPF).to(self.device)
            

        #self.phi0_BPFL = g_prod.mean().item()
        self.phi0_BPFL = g_prod
        print('BPF done with', self.K_BPF_large, 'particles', 'Z:',self.phi0_BPFL, 'log Z:', np.log(self.phi0_BPFL))
        #dont use this, use self.Ztrue later
        
        
        
        '''
        before training, calculate some constants using larger MC sample size
        '''

        KK = 100000
        X_KK = self.X_0.repeat(KK, 1).to(self.device)

        y_observed_KK = []#define y_OB for KK particles
        for n in range(self.N + 1):
            y_observed_KK.append(self.y_observed[n].repeat(KK,1).to(self.device))
            

        Z_sum_forphi0 = pt.zeros(KK).to(self.device)

        Z_sum_forphi0 += (-self.f(X_KK - y_observed_KK[0], 0))
        
        for n in range(self.N):
            X_KK = X_KK + self.delta_t * self.b(X_KK) + self.sq_delta_t * pt.randn_like(X_KK)
            Z_sum_forphi0 += -self.f(X_KK - y_observed_KK[n+1], (n+1) * self.delta_t)
        phi0 = (pt.exp(Z_sum_forphi0)).mean().item()
        #entropy_inCE = ((Z_sum_forphi0 - np.log(phi0)) * pt.exp(Z_sum_forphi0)).mean().item() / phi0
        entropy_inCE = ((Z_sum_forphi0 - np.log(self.Z_true)) * pt.exp(Z_sum_forphi0)).mean().item() / self.Z_true
        print('constants calculated with', KK, 'samples, without resampling')
        print('phi_0^*:', phi0, 'entropy_inCE:', entropy_inCE)
        print('logphi_0^*:',np.log(phi0))
        #print((sdfsfsf))
        
        
       
        y_observed_K = []
        for n in range(self.N + 1):
            y_observed_K.append(self.y_observed[n].repeat(self.K,1).to(self.device))    


        '''
        simulate the loss function (along the path in time)
        '''
        print('N',self.N,'T',self.T,'delta_t',self.delta_t.item())
        for l in range(self.L):
            if(l % self.print_every == 0):
                print('iteration',l,end=' ')
            t_0 = time.time()

            X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi = self.initialize_training_data()
            additional_loss = pt.zeros(self.K)

            
            X_untwisted = self.X_0.repeat(self.K, 1).to(self.device)

            
            #Z_sum_forloss = pt.zeros(self.K).to(self.device)
            Z_sum_forCE1 = pt.zeros(self.K).to(self.device)
            Z_sum_forCE2 = pt.zeros(self.K).to(self.device)

            Z_sum_forRE_untwisted1 = pt.zeros(self.K).to(self.device)
            Z_sum_forRE_untwisted2 = pt.zeros(self.K).to(self.device)

            #Z_sum += self.f(X-y_observed_K[0],0)
            Z_sum_forRE_untwisted1 += self.f(X_untwisted-y_observed_K[0],0)
            Z_sum_forCE2 += -self.f(X_untwisted-y_observed_K[0],0)
            

            X_directed = self.X_0.repeat(self.K, 1).to(self.device)

            Z_sum_forRE_direct_1 = pt.zeros(self.K).to(self.device)
            Z_sum_forRE_direct_2 = pt.zeros(self.K).to(self.device)
            Z_sum_forRE_direct_22 = pt.zeros(self.K).to(self.device)

            Z_sum_forRE_direct_1 += self.f(X_directed - y_observed_K[0], 0)
            #logg0 = -self.f(pt.zeros(self.d).repeat(self.K, 1).to(self.device) - y_observed_K[0], 0)
            
            for n in range(self.N):
                #print(pt.cuda.memory_allocated()/1024**2)
                #print('------------------------------')
    
                if n==40:
                    if(l % self.print_every == 0):
                        
                        print('phi_40:', self.mean_z_n(pt.tensor([[n]],dtype=pt.float32).to(self.device)),end=' ')
                       
                        print(self.var_z_n(pt.tensor([[n]],dtype=pt.float32).to(self.device)),end=' ')
                        print(' ')
                        

                '''
                simulate J(u) using monte carlo,
                
                '''
 
                if(self.sampling_method == 'untwisted'):
                    #first calculate the normalizing constant with the old X,X_untwisted, denote by repa_sumA,repa_sumA_untwisted
                    tildeN = 50
                    
                    repa_sumA_untwisted = pt.zeros_like(Z).to(self.device)
                    #noise = pt.randn(X_untwisted.shape[0],tildeN).to(self.device)

                    
                    for repa_i in range(tildeN):
                        
                        repara_Ui_untwisted = self.sq_delta_t * pt.randn_like(X_untwisted).to(self.device) + X_untwisted + self.delta_t * self.b(X_untwisted)

                        repa_phii_untwisted = pt.exp(self.Z_n_(repara_Ui_untwisted,n+1))
                        repa_sumA_untwisted += repa_phii_untwisted
 
                    repa_sumA_untwisted /= tildeN#tilde phi for untwisted model

                    #move forward
                    #update untwisted Markov Chain
                    
                    X_untwisted = X_untwisted + self.delta_t * self.b(X_untwisted) + self.sq_delta_t * pt.randn_like(X_untwisted).to(self.device)
                    

                if(self.sampling_method == 'direct'):
                    
                    #update Markov chain
                    X_directed_new = (self.delta_t * self.mean_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)).expand(X_directed.shape[0],-1)\
                                   + self.var_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)) * (X_directed + self.delta_t * self.b(X_directed)))\
                          / (self.delta_t + self.var_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)))\
                              + pt.sqrt((self.var_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)) * self.delta_t)\
                                         / (self.var_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)) + self.delta_t))\
                                  * pt.randn_like(X_directed).to(self.device)
                    
                    X_untwisted_new = X_untwisted + self.delta_t * self.b(X_untwisted) + self.sq_delta_t * pt.randn_like(X_untwisted).to(self.device)
                    

                    #calculate loss_RE
                    Z_sum_forRE_direct_2 += (self.log_phi_Z_n_(X_directed_new, n+1) - self.log_tildephi_Z_n_(X_directed, n)).squeeze()    
                                 
                    #Z_sum_forRE_direct_2 += (self.log_phi_Z_n_scale(X_directed_new, n+1) - self.log_tildephi_Z_n_scale(X_directed, n)).squeeze()
                    Z_sum_forRE_direct_22 += (self.log_phi_Z_n_(X_directed_new.detach(), n+1)\
                                               - self.log_tildephi_Z_n_(X_directed.detach(), n)).squeeze()
                    Z_sum_forRE_direct_1 += self.f(X_directed_new - y_observed_K[n+1], (n+1) * self.delta_t)
                    # print('logphi-logtildephi',(self.log_phi_Z_n_(X_directed_new, n+1) - self.log_tildephi_Z_n_(X_directed, n)).squeeze()[0])
                    # print('X',X_directed[0])
                    # print('-loggk(X)',self.f(X_directed - y_observed_K[n+1], (n+1) * self.delta_t)[0])
                    #print(sdfsfdsa)

                    #calculate loss_CE
                    Z_sum_forCE1 += self.log_phi_Z_n_(X_untwisted_new, n+1).squeeze() - self.log_tildephi_Z_n_(X_untwisted, n).squeeze() 
                    Z_sum_forCE2 += -self.f(X_untwisted_new - y_observed_K[n+1], (n+1)*self.delta_t) 
                    


                    #calculate loss_RE with X_untwisted
                    # Z_sum_forRE_untwisted1 += self.f(X_untwisted_new - y_observed_K[n+1], (n+1) * self.delta_t)
                    # Z_sum_forRE_untwisted2 += self.log_phi_Z_n_(X_untwisted_new, n+1).squeeze() - self.log_tildephi_Z_n_(X_untwisted, n).squeeze() 
                    # print(Z_sum_forRE_untwisted1, Z_sum_forRE_untwisted2)

                    X_directed = X_directed_new
                    X_untwisted = X_untwisted_new


                '''
                calculate Z_sum i.e. the current path integral 
                '''
                if 'relative_entropy' in self.loss_method:

                    '''
                    loss = E[sum(log g_k(X_n^varphi)) + sum(log varphi_k(X_k^varphi)) - sum(log tilde_varphi(X_k-1^varphi))]
                    '''
                    if(self.sampling_method=='rejectsampling'):
                        Z_sum += self.f(X, (n+1) * self.delta_t) + self.Z_n_(X, n+1).squeeze() - pt.log(repa_sumA).squeeze()
                        Z_sum_forloss += self.Z_n_(X,n+1).squeeze() - pt.log(repa_sumA).squeeze()
                    #here, f = -log g_k
                    # Z_sum_forCE1 += self.Z_n_(X_untwisted,n+1).squeeze() - pt.log(repa_sumA_untwisted).squeeze()
                    # Z_sum_forCE2 += -self.f(X_untwisted-y_observed_K[n+1],(n+1)*self.delta_t) 
                                        
                    # Z_sum_forRE_untwisted1 += self.f(X_untwisted-y_observed_K[n+1],(n+1)*self.delta_t) + self.Z_n_(X_untwisted,n+1).squeeze() - pt.log(repa_sumA_untwisted).squeeze()
                    # Z_sum_forRE_untwisted2 += self.Z_n_(X_untwisted,n+1).squeeze() - pt.log(repa_sumA_untwisted).squeeze()

                    pt.cuda.empty_cache()
                    # print('------------------------')
                    # print(pt.cuda.memory_allocated()/1024**2)

                    #print("Z_sum calculated!")

            '''
            update parameters and loss (auto-calculate gradient)
            '''
            if(self.sampling_method=='reparameterize'):
                loss = self.gradient_descent(X, Y, Z_sum, l, additional_loss.mean())
                print('l',l,'loss',loss.item())
            else:
                
               
                if(self.sampling_method=='rejectsampling'):
                    loss_RE = Z_sum.mean() + pt.tensor(np.log(phi0))
                    print('iteration',l,'average reject times:', averagetry/self.K,'REloss:',loss_RE.item(),end=' ')

                else:
                    # print('Zsum1',Z_sum_forRE_direct_1)
                    # print('Zsum2',Z_sum_forRE_direct_2)
                    loss_RE = (Z_sum_forRE_direct_1 + Z_sum_forRE_direct_2).mean() + pt.tensor(np.log(self.Z_true)).to(self.device)
                    loss_RE_for_grad = ((1 + Z_sum_forRE_direct_2 + Z_sum_forRE_direct_1).detach() * Z_sum_forRE_direct_22).mean()

                    #loss_RE = (pt.exp(Z_sum_forRE_untwisted2) * (Z_sum_forRE_untwisted1 )).mean() + pt.tensor(np.log(self.phi0_BPFL)).to(self.device)
                    #loss_RE = (pt.exp(Z_sum_forRE_untwisted2) * (Z_sum_forRE_untwisted1 )).mean() + pt.tensor(np.log(self.Z_true)).to(self.device)
                    if(l % self.print_every == 0):
                        print('REloss:',loss_RE.item(),end=' ')

                #loss_CE = -(pt.exp(Z_sum_forCE2) * Z_sum_forCE1).mean()  + entropy_inCE 
                #loss_CE = -(pt.exp(Z_sum_forCE2 - pt.tensor(np.log(self.phi0_BPFL)).to(self.device)) * Z_sum_forCE1).mean()

                    loss_CE = -(pt.exp(Z_sum_forCE2 - pt.tensor(np.log(self.Z_true)).to(self.device)) * Z_sum_forCE1).mean() + entropy_inCE
                    if(l % self.print_every == 0):
                        print('CEloss:',loss_CE.item(),end=' ')

                    loss_RECE = loss_CE + loss_RE
                    if(l % self.print_every == 0):
                        print('RECEloss:',loss_RECE.item(),end=' ')

                    loss_RelVar = (pt.exp(2 * Z_sum_forCE2 - Z_sum_forCE1)).mean()
                    if(l % self.print_every == 0):
                        print('RelVar:',loss_RelVar.item() - self.Z_true**2,end=' ')


                    # loss_RE_untwisted = (pt.exp(Z_sum_forRE_untwisted2) * (Z_sum_forRE_untwisted1 + Z_sum_forRE_untwisted2)).mean()\
                    #       + pt.tensor(np.log(self.Z_true)).to(self.device)
                    # if(l % self.print_every == 0):
                    #     print('REloss(untwisted):',loss_RE_untwisted.item(),end=' ')
                
                # if(l % self.print_every == 0):
                #     print('-CEloss:',loss_CE.item(),end=' ')

                # loss_CERE = loss_RE + loss_CE
                # if(l % self.print_every == 0):
                #     print('CEREloss:',loss_CERE.item(),end = ' ')


                tt1 = time.time()
                #print(self.train_goal)
                if(self.train_goal =='REdirect'):
                    self.zero_grad()
                    #loss_RE_for_grad.backward()
                    loss_RE.backward()
                    self.optimization_step()

                elif(self.train_goal == 'CEdirect'):
                    self.zero_grad()
                    loss_CE.backward()
                    self.optimization_step()

                elif(self.train_goal == 'RECEdirect'):
                    self.zero_grad()
                    loss_RECE.backward()
                    self.optimization_step()

                elif(self.train_goal == 'REmanualloss'):
                    
              
                    self.zero_grad()
                    loss_RE_for_grad.backward()
                    self.optimization_step()

                elif(self.train_goal == 'CEREtwisted'):
                    #trainCERE twisted
                    trueloss_CERE = ((Z_sum.detach() + 1) * Z_sum_forloss).mean() + loss_CE
                    self.zero_grad()
                    trueloss_CERE.backward()
                    self.optimization_step()

                elif(self.train_goal == 'CE'):
                    #train CE
                    trueloss_CE = loss_CE
                    self.zero_grad()
                    trueloss_CE.backward()
                    self.optimization_step()

                elif(self.train_goal == 'CERE'):
                    #train CERE untwisted
                    trueloss_CERE_untwisted = loss_CERE
                    self.zero_grad()
                    trueloss_CERE_untwisted.backward()
                    self.optimization_step()

                elif(self.train_goal == 'RE'):
                    #train RE untwisted
                    trueloss_RE_untwisted = loss_RE
                    self.zero_grad()
                    trueloss_RE_untwisted.backward()
                    self.optimization_step()

                else:
                    print('invalid loss choice!')


                tt2=time.time()
                if(l % self.print_every == 0):
                    print('training time:',tt2-tt1,end=' ')
                # print(pt.cuda.memory_allocated(self.device)/1024**2)
                # print(pt.cuda.memory_reserved(self.device)/1024**2)
                

            if self.log_gradient is True:
                grads = [params.grad for params in list(filter(lambda params: params.requires_grad, self.z_n.parameters()))
                        if params.grad is not None]
                grads_flat = pt.zeros(self.p)        
                i = 0
                for grad in grads:
                    grad_flat = grad.reshape(-1)
                    j = len(grad_flat)
                    grads_flat[i:i + j] = grad_flat
                    i += j

                self.gradient_log[l, :] = grads_flat.cpu().detach()

            #self.loss_log_RE_untwisted.append(loss_RE_untwisted.item())
            self.loss_log_CERE.append(loss_RECE.item())
            self.loss_log_CE.append(loss_CE.item())
            self.relvar_log.append(loss_RelVar.item())

            self.loss_log_REdirect.append(loss_RE.item())
            

            t_1 = time.time()
            self.times.append(t_1 - t_0)
            if(l % self.print_every == 0):
                print('running time:',t_1 - t_0)



        print("training ended!")


        '''
        finally run a TPF to calculate Z with same particle numbers of BPF (K_BPF_small)
        for self.replciates_num times
        '''
        
        ESS_ave = 0
        for ii in range(self.replicate_num):
        #for ii in range(0):
            ithESSnum = 0

            t_TPF1 = time.time()
                
            X_TPF = self.X_0.repeat(self.K_BPF_small, 1).to(self.device)
                

            y_observed_KBPFL = []#define y_OB for K_BPF_large particles
            for n in range(self.N + 1):
                y_observed_KBPFL.append(self.y_observed[n].repeat(self.K_BPF_small,1).to(self.device))

            g_prod = 1
            #log_grod = 0
            #g_prod = pt.ones(models[0].K_BPF_large).to(models[0].device)
            for n in range(self.N+1):
                # print('----------------------------')
                # print('n=',n)
                g_k = pt.exp(-self.f(X_TPF - y_observed_KBPFL[n], n * self.delta_t))
                #logg_k = -models[0].f(X_TPF - y_observed_KBPFL[n], n * models[0].delta_t)
            
                if(n == self.N):
                    gphi_k = g_k / self.phi_Z_n_(X_TPF,n).squeeze()
                    #log_gphi_k = logg_k - models[0].log_phi_Z_n_(X_TPF,n).squeeze()
                    
                elif(n == 0):
                    # gphi_k = g_k * tildephi_TPF.squeeze()
                    gphi_k = g_k * self.tildephi_Z_n_(X_TPF, n).squeeze()
                    #log_gphi_k = logg_k + models[0].log_tildephi_Z_n_(X_TPF, n).squeeze()
                    #print('tildephik',models[0].tildephi_Z_n_(X_TPF, n).squeeze().shape)

                    

                else:
                    # gphi_k = g_k * tildephi_TPF.squeeze() / pt.exp(self.Z_n_(X_TPF, n)).squeeze()
                    
                    gphi_k = g_k * self.tildephi_Z_n_(X_TPF,n).squeeze() / self.phi_Z_n_(X_TPF,n).squeeze()
                    #log_gphi_k = logg_k + models[0].log_tildephi_Z_n_(X_TPF,n).squeeze() - models[0].log_tildephi_Z_n_(X_TPF,n).squeeze()
                    # print('tildephi', self.tildephi_Z_n_(X_TPF,n).squeeze()[0])
                    # print('phi', self.phi_Z_n_(X_TPF,n).squeeze()[0])
                    # print('tildephi/phi',(self.tildephi_Z_n_(X_TPF,n).squeeze() / self.phi_Z_n_(X_TPF,n).squeeze())[0] )

                
                #g_prod *= pt.exp(log_gphi_k).mean().item()
                #print(gphi_k.mean().item())
                g_prod *= gphi_k.mean().item()
                #print(g_prod)
                    
                    
                #resample
                weights = gphi_k / pt.sum(gphi_k)
                #weights = pt.exp(log_gphi_k) / pt.sum(pt.exp(log_gphi_k))
                indices = pt.multinomial(weights, self.K_BPF_small, replacement=True)
                ESS = 1.0 / np.sum((weights.detach().to('cpu').numpy())**2)
                ESS_ave += ESS

                if(ESS <= self.resample_kappa * self.K_BPF_small):
                    X_TPF = X_TPF[indices]
                    self.average_ESSnum += 1
                    ithESSnum += 1
                    
                
                # print(pt.cuda.memory_allocated(self.device)/1024**2)
                # print(pt.cuda.memory_reserved(self.device)/1024**2)
                #print(X_TPF.shape)
                X_TPF = (self.delta_t * self.mean_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)).expand(X_TPF.shape[0],-1)\
                                + self.var_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)) * (X_TPF + self.delta_t * self.b(X_TPF)))\
                        / (self.delta_t + self.var_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)))\
                            + pt.sqrt((self.var_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)) * self.delta_t)\
                                        / (self.var_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)) + self.delta_t))\
                                * pt.randn_like(X_TPF).to(self.device)
                # print('X',X_TPF[0])
                # print('mean',self.mean_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)))
                # print('var',self.var_z_n(pt.tensor([[n+1]],dtype=pt.float32).to(self.device)))
                pt.cuda.empty_cache()
                # print(pt.cuda.memory_allocated(self.device)/1024**2)
                # print(pt.cuda.memory_reserved(self.device)/1024**2)
                #print(X_TPF)
                #print(indices)

                        

            
            t_TPF2 = time.time()
            print('1 time TPF ended in time', t_TPF2 - t_TPF1, 'Z = ', g_prod, 'log Z =', np.log(g_prod), 'ESS number:', ithESSnum)
            self.TPF_Z.append(g_prod)
            self.TPF_logZ.append(np.log(g_prod))

            self.average_ESSnum /= self.replicate_num
            print('average ESS number:', self.average_ESSnum)

        
        ESS_ave /= (self.replicate_num * (self.N + 1))
        print('ESS_ave=', ESS_ave, 'ESS_ave percentage=', ESS_ave / self.K_BPF_small)
        










class mySVSolver():

    def __init__(self, name, problem, lr=0.01, L=500, K=500, delta_t=0.25,
                 approx_method='control', loss_method='relative-entropy', time_approx='outer',
                 learn_Y_0=False, adaptive_forward_process=True, detach_forward=False,
                 early_stopping_time=10000, random_X_0=False, compute_gradient_variance=0,
                 IS_variance_K=0, IS_variance_iter=1, metastability_logs=None, print_every=100, plot_trajectories=None,
                 seed=42, save_results=False, u_l2_error_flag=True, log_gradient=False,  verbose=False, T = 0.5,
                 sampling_method = 'untwisted', train_goal = 'RE', d = 2, OBsigma2 = 1, y_observed = [],
                 K_BPF_large = 10000, K_BPF_small = 200, phi0_BPF = 1, replicate_num = 1000, Z_true = 1, alpha = 3.0,standard = 300):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = d
        self.T = T
        # self.X_0 = problem.X_0
        self.X_0 = pt.ones(self.d) + 0
        self.Y_0 = pt.tensor([0.0])
        self.X_u_opt = None
        

        self.OBsigma2 = OBsigma2
        self.y_observed = y_observed

        self.Z_true = Z_true

        self.alpha = alpha
        

        self.sampling_method = sampling_method
        self.train_goal = train_goal

        # hyperparameters
        self.device = pt.device('cuda')
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(self.device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(self.device)
        self.N = int(np.floor(self.T / self.delta_t_np)) # number of steps
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size
        self.random_X_0 = random_X_0

        self.K_BPF_large = K_BPF_large#particle number for BPF
        self.K_BPF_small = K_BPF_small
        self.phi0_BPFL = phi0_BPF
        self.replicate_num = replicate_num

        self.standard = standard


        # learning properties
        self.loss_method = loss_method
        self.approx_method = approx_method
        self.learn_Y_0 = learn_Y_0
        self.adaptive_forward_process = adaptive_forward_process
        self.detach_forward = detach_forward
        self.early_stopping_time = early_stopping_time
        

        self.has_ref_solution = hasattr(problem, 'u_true')
        self.u_l2_error_flag = u_l2_error_flag
        if self.has_ref_solution is False:
            self.u_l2_error_flag = False

        if self.loss_method == 'relative_entropy':
            self.adaptive_forward_process = True
        if self.loss_method == 'cross_entropy':
            self.learn_Y_0 = False

        # printing and logging
        self.print_every = print_every
        self.verbose = verbose
        self.verbose_NN = False
        self.save_results = save_results
        self.compute_gradient_variance = compute_gradient_variance
        self.IS_variance_K = IS_variance_K
        self.IS_variance_iter = IS_variance_iter
        self.metastability_logs = metastability_logs
        self.plot_trajectories = plot_trajectories
        self.metastability_logs = metastability_logs
        self.log_gradient = log_gradient
        self.print_gradient_norm = False

        # function approximation
        self.Phis = []
        self.time_approx = time_approx

        pt.manual_seed(seed)
        if self.approx_method == 'control':
            '''
            define NN here: at each time i z_n[i] = NN[i] from R^d to R^d
            or NN = NN(t,x) 
            output of NN is changed into 1d
            '''
            if self.time_approx == 'outer':
                self.z_n = [DenseNet(d_in=self.d, d_out=self.d, lr=self.lr, seed=seed) for i in range(self.N)]
                #self.z_n = [DenseNet_tanh(d_in=self.d, d_out=1, lr=self.lr, seed=seed) for i in range(self.N)]
            elif self.time_approx == 'inner':
                self.z_n = DenseNet(d_in=self.d + 1, d_out=1, lr=self.lr, seed=120)
                #self.z_n = MySequential(d_in=self.d + 1, d_out=self.d, lr=self.lr, seed=123)
                #self.z_n = MySequential(d_in=self.d + 1, d_out=1, lr=self.lr, seed=123)


        self.update_Phis()

        for phi in self.Phis:
            phi.train()

        if self.verbose_NN is True:
            if self.time_approx == 'outer':
                print('%d NNs, %d parameters in each network, total parameters = %d'
                      % (self.N, self.p, self.p * self.N))
            else:
                print('%d NNs, %d parameters in each network, total parameters = %d'
                      % (1, self.p, self.p))

        # logging
        self.Y_0_log = []

        self.loss_log = []
        self.loss_log_CE = []
        self.loss_log_RE = []
        self.loss_log_CERE = []
        self.relvar_log = []

        self.TPF_Z = []
        self.TPF_logZ = []


        self.u_L2_loss = []
        self.IS_rel_log = []
        self.times = []
        self.grads_rel_error_log = []
        self.particles_close_to_target = []

    def b(self, x):
        bi, bj = pt.meshgrid(pt.arange(self.d), pt.arange(self.d))
        #alpha = 0.5
        bA = self.alpha ** (pt.abs(bi - bj) + 1)
        bA = bA.to(self.device)

        # bbeta = 0.1
        # bA = pt.eye(d).to(device) 
        # bA[-1, -1] = bbeta

        #return (pt.mm(bA, x.t()).t() - x) / self.delta_t
        #return 0
        return -pt.mm(bA, x.t()).t()
        #return -0.5* x

    def sigma(self, x):
        return self.problem.sigma(x)

    def h(self, t, x, y, z):
        return self.problem.h(t, x, y, z)

    def f(self, xx, yy, t):
        diag_elements = pt.exp(xx)
        exponent = -0.5 * pt.sum(yy ** 2 / diag_elements, dim=1)
        normalization_constant = pt.sqrt((2 * pt.pi) ** self.d * pt.prod(diag_elements, dim=1))
        #normalization_constant = 1
        density = pt.exp(exponent) / normalization_constant
        return -pt.log(density)

    def g(self, x):
        return self.problem.g(x)

    def u_true(self, x, t):
        return self.problem.u_true(x, t)

    def v_true(self, x, t):
        return self.problem.v_true(x, t)

    def update_Phis(self):
        if self.approx_method == 'control':
            if self.learn_Y_0 is True:
                if self.time_approx == 'outer':
                    self.Phis = self.z_n + [self.y_0]
                elif self.time_approx == 'inner':
                    self.Phis = [self.z_n, self.y_0]
            else:
                '''
                outer means t in outside NN
                inner means NN = NN(t,x)
                '''
                if self.time_approx == 'outer':
                    self.Phis = self.z_n
                elif self.time_approx == 'inner':
                    self.Phis = [self.z_n]
        for phi in self.Phis:
            phi.to(self.device)
        '''
        number of the parameters 
        '''
        self.p = sum([np.prod(params.size()) for params in
          filter(lambda params: params.requires_grad,
                 self.Phis[0].parameters())])
        if self.log_gradient is True:
            self.gradient_log = pt.zeros(self.L, self.p)

    def loss_function(self, X, Y, Z_sum, l):
        if self.loss_method == 'relative_entropy':
            #return (Z_sum + self.g(X)).mean() #changed into particle filter's setting, g is moving inside Z_sum
            return (Z_sum ).mean()
      

    def zero_grad(self):
        for phi in self.Phis:
            phi.optim.zero_grad()

    def optimization_step(self):
        for phi in self.Phis:
            phi.optim.step()

    def gradient_descent(self, X, Y, Z_sum, l, additional_loss):
        self.zero_grad()

        loss = self.loss_function(X, Y, Z_sum, l) + additional_loss
        loss.backward()
        self.optimization_step()
        return loss

    def flatten_gradient(self, k, grads, grads_flat):
        i = 0
        for grad in grads:
            grad_flat = grad.reshape(-1)
            j = len(grad_flat)
            grads_flat[k, i:i + j] = grad_flat
            i += j
        return grads_flat

    def get_gradient_variances(self, X, Y):
        grads_mean = pt.zeros(self.N, self.p)
        grads_var = pt.zeros(self.N, self.p)

        for n in range(self.N):

            grads_Y_flat = pt.zeros(self.K, self.p)

            for k in range(self.K):
                self.zero_grad()
                Y[k].backward(retain_graph=True)

                grad_Y = [params.grad for params in list(filter(lambda params:
                                                                params.requires_grad,
                                                                self.z_n[n].parameters()))
                          if params.grad is not None]

                grads_Y_flat = self.flatten_gradient(k, grad_Y, grads_Y_flat)

            grads_g_X_flat = pt.zeros(self.K, self.p)

            if self.adaptive_forward_process is True:

                for k in range(self.K):
                    self.zero_grad()
                    self.g(X[0, :].unsqueeze(0)).backward(retain_graph=True)

                    grad_g_X = [params.grad for params in list(filter(lambda params:
                                                                      params.requires_grad,
                                                                      self.z_n[n].parameters()))
                                if params.grad is not None]

                    grads_g_X_flat = self.flatten_gradient(k, grad_g_X, grads_g_X_flat)

            if self.loss_method == 'moment':
                grads_flat = 2 * (Y - self.g(X)).unsqueeze(1) * (grads_Y_flat - grads_g_X_flat)
            elif self.loss_method == 'log-variance':
                grads_flat = 2 * (((Y - self.g(X)).unsqueeze(1)
                                   - pt.mean((Y - self.g(X)).unsqueeze(1), 0).unsqueeze(0))
                                  * (grads_Y_flat - grads_g_X_flat
                                     - pt.mean(grads_Y_flat - grads_g_X_flat, 0).unsqueeze(0)))

            grads_mean[n, :] = pt.mean(grads_flat, dim=0)
            grads_var[n, :] = pt.var(grads_flat, dim=0)

        grads_rel_error = pt.sqrt(grads_var) / grads_mean
        grads_rel_error[grads_rel_error != grads_rel_error] = 0
        return grads_rel_error

    def state_dict_to_list(self, sd):
        sd_list = {}
        for name in sd:
            if type(sd[name]) == pt.Tensor:
                sd_list[name] = sd[name].detach().cpu().numpy().tolist()
            else:
                sd_list[name] = sd[name]
        return sd_list

    def list_to_state_dict(self, l):
         return {param: pt.tensor(l[param]) if type(l[param]) == list else l[param] for param in l}

    def save_logs(self, model_name='model'):
        # currently does not work for all modi
        logs = {'name': self.name, 'date': self.date, 'd': self.d, 'T': self.T,
                'seed': self.seed, 'delta_t': self.delta_t_np, 'N': self.N, 'lr': self.lr,
                'K': self.K, 'loss_method': self.loss_method, 'learn_Y_0': self.learn_Y_0,
                'adaptive_forward_process': self.adaptive_forward_process,
                'Y_0_log': self.Y_0_log, 'loss_log': self.loss_log, 'u_L2_loss': self.u_L2_loss,
                'Phis_state_dict': [self.state_dict_to_list(z.cpu().state_dict()) for z in self.Phis]}

        path_name = 'logs/%s_%s_%s.json' % (model_name, self.name, self.date)
        i = 1
        while os.path.isfile(path_name):
            i += 1
            path_name = 'logs/%s_%s_%s_%d.json' % (model_name, self.name, self.date, i)

        with open(path_name, 'w') as f:
            json.dump(logs, f, indent=2)

    def save_networks(self):
        data_dict = {}
        idx = 0
        for z in self.Phis:
            key = 'nn%d' % idx
            data_dict[key] = z.state_dict()
            idx += 1
        path_name = 'output/%s_%s.pt' % (self.name, self.date)
        pt.save(data_dict, path_name)
        print('\nnetworks data has been stored to file: %s' % path_name)

    def load_networks(self, cp_name):
        print('\nload network data from file: %s' % cp_name)
        checkpoint = pt.load(cp_name)
        idx = 0
        for z in self.Phis:
            key = 'nn%d' % idx
            z.load_state_dict(checkpoint[key])
            z.eval()
            idx += 1

    def compute_grad_Y(self, X, n):
        Y_n_eval = self.Y_n(X, n).squeeze(1).sum()
        Y_n_eval.backward(retain_graph=True)
        Z, = pt.autograd.grad(Y_n_eval, X, create_graph=True)
        Z = pt.mm(self.sigma(X), Z.t()).t()
        return Z

    def Y_n(self, X, t):
        n = int(np.ceil(t / self.delta_t))
        if self.time_approx == 'outer':
            return self.y_n[n](X)
        elif self.time_approx == 'inner':
            t_X = pt.cat([pt.ones([X.shape[0], 1]) * t, X], 1)
            return self.y_n[0](t_X)

    def Z_n_(self, X, n):
        if self.approx_method == 'control':
            if self.time_approx == 'outer':
                n = max(0, min(n, self.N - 1))
                return self.z_n[n](X)
            elif self.time_approx == 'inner':
                t_X = pt.cat([pt.ones([X.shape[0], 1]).to(self.device) * n * self.delta_t, X], 1)
                return self.z_n(t_X)

    def Z_n(self, X, t):
        n = int(pt.ceil(t / self.delta_t))
        return self.Z_n_(X, n)

    '''
    initialize
    '''
    def initialize_training_data(self):
        X = self.X_0.repeat(self.K, 1).to(self.device)
        if self.random_X_0 is True:
            X = pt.randn(self.K, self.d).to(self.device)
        Y = self.Y_0.repeat(self.K).to(self.device)
        if self.approx_method == 'value_function':
            X = pt.autograd.Variable(X, requires_grad=True)
            Y = self.Y_n(X, 0)[:, 0]
        elif self.learn_Y_0 is True:
            Y = self.y_0(X)
            self.Y_0_log.append(Y[0].item())
        Z_sum = pt.zeros(self.K).to(self.device)
        u_L2 = pt.zeros(self.K).to(self.device)
        u_int = pt.zeros(self.K).to(self.device)
        u_W_int = pt.zeros(self.K).to(self.device)
        double_int = pt.zeros(self.K).to(self.device)

        xi = pt.randn(self.K, self.d, self.N + 1).to(self.device)
        return X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi

    def train_LSE_with_reference(self):
        if self.approx_method != 'control':
            print('only learn control with reference solution!')
        if self.has_ref_solution == False:
            print('reference solution is needed!')

        print('\nd = %d, L = %d, K = %d, delta_t = %.2e, N = %d, lr = %.2e, %s, %s, %s, %s\n' % (self.d, self.L, self.K, self.delta_t_np, self.N, self.lr, self.approx_method, self.time_approx, self.loss_method, 'adaptive' if self.adaptive_forward_process else ''))

        xb = 2.0
        X = pt.linspace(-xb, xb, 200).unsqueeze(1)

        for l in range(self.L):
            t_0 = time.time()
            loss = 0.0
            for n in range(self.N):
                loss += pt.sum((- self.Z_n_(X, n) - pt.tensor(self.u_true(X, n * self.delta_t_np)).float())**2) * self.delta_t
            self.zero_grad()
            loss.backward()
            self.optimization_step()

            self.loss_log.append(loss.item())

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                string = ('%d - loss: %.3e - time/iter: %.2fs' % (l, self.loss_log[-1],
                          np.mean(self.times[-self.print_every:])))
                print(string + ' - gradient l_inf: %.3e' %
                      (np.array([max([pt.norm(params.grad.data, float('inf')).item() for params
                                      in filter(lambda params: params.requires_grad,
                                                phi.parameters())])
                                 for phi in self.Phis]).max()))

        #   self.save_networks()

    def train(self):

        pt.manual_seed(self.seed)


        
       
        y_observed_K = []
        for n in range(self.N + 1):
            y_observed_K.append(self.y_observed[n].repeat(self.K,1).to(self.device))    


        '''
        simulate the loss function (along the path in time)
        '''
        print('N',self.N,'T',self.T,'delta_t',self.delta_t.item())
        for l in range(self.L):
            if(l % self.print_every == 0):
                print('iteration',l,end=' ')
            t_0 = time.time()

            X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi = self.initialize_training_data()
            additional_loss = pt.zeros(self.K)

            
            X_untwisted = self.X_0.repeat(self.K, 1).to(self.device)

            
            Z_sum_forloss = pt.zeros(self.K).to(self.device)
            Z_sum_forCE1 = pt.zeros(self.K).to(self.device)
            Z_sum_forCE2 = pt.zeros(self.K).to(self.device)

            Z_sum_forRE_untwisted1 = pt.zeros(self.K).to(self.device)
            Z_sum_forRE_untwisted2 = pt.zeros(self.K).to(self.device)

            Z_sum += self.f(X,y_observed_K[0],0)
            Z_sum_forRE_untwisted1 += self.f(X,y_observed_K[0],0)
            Z_sum_forCE2 += -self.f(X,y_observed_K[0],0)

              
 
            for n in range(self.N):
                #print(pt.cuda.memory_allocated()/1024**2)
                Z = self.Z_n_(X_untwisted, n)
                if n==2:
                    if(l % self.print_every == 0):
                        print('log phi_2:',Z.squeeze().mean().item(),end=' ')

                # c = pt.zeros(self.d, self.K).to(self.device)
                # if self.adaptive_forward_process is True:
                #     c = -self.Z_n_(X, n).t()
                #     #c = -Z.t()


                if self.detach_forward is True or (self.loss_method == 'log-variance-repa' and l % 2 == 1):
                    c = c.detach()


                '''
                simulate J(u) using monte carlo,
                
                '''

                if(self.sampling_method == 'untwisted'):
                    #first calculate the normalizing constant with the old X,X_untwisted, denote by repa_sumA,repa_sumA_untwisted
                    tildeN = 150
                    
                    repa_sumA_untwisted = pt.zeros_like(Z).to(self.device)
                    #noise = pt.randn(X_untwisted.shape[0],tildeN).to(self.device)

                    
                    for repa_i in range(tildeN):
                        
                        repara_Ui_untwisted = self.sq_delta_t * pt.randn_like(X_untwisted).to(self.device) + X_untwisted + self.delta_t * self.b(X_untwisted)

                        repa_phii_untwisted = pt.exp(self.Z_n_(repara_Ui_untwisted,n+1))
                        repa_sumA_untwisted += repa_phii_untwisted
 
                    repa_sumA_untwisted /= tildeN#tilde phi for untwisted model

                    #move forward
                    #update untwisted Markov Chain
                    
                    X_untwisted = X_untwisted + self.delta_t * self.b(X_untwisted) + self.sq_delta_t * pt.randn_like(X_untwisted).to(self.device)
                    
                '''
                calculate Z_sum i.e. the current path integral 
                '''
                if 'relative_entropy' in self.loss_method:

                    '''
                    loss = E[sum(log g_k(X_n^varphi)) + sum(log varphi_k(X_k^varphi)) - sum(log tilde_varphi(X_k-1^varphi))]
                    '''
                    
                    #here, f = -log g_k
                    Z_sum_forCE1 += self.Z_n_(X_untwisted,n+1).squeeze() - pt.log(repa_sumA_untwisted).squeeze()
                    Z_sum_forCE2 += -self.f(X_untwisted,y_observed_K[n+1],(n+1)*self.delta_t) 
                    
                    
                    Z_sum_forRE_untwisted1 += self.f(X_untwisted,y_observed_K[n+1],(n+1)*self.delta_t) + self.Z_n_(X_untwisted,n+1).squeeze() - pt.log(repa_sumA_untwisted).squeeze()
                    Z_sum_forRE_untwisted2 += self.Z_n_(X_untwisted,n+1).squeeze() - pt.log(repa_sumA_untwisted).squeeze()
                    

                    pt.cuda.empty_cache()
                    # print('------------------------')
                    # print(pt.cuda.memory_allocated()/1024**2)

                    #print("Z_sum calculated!")



            '''
            update parameters and loss (auto-calculate gradient)
            '''
            if(self.sampling_method=='reparameterize'):
                loss = self.gradient_descent(X, Y, Z_sum, l, additional_loss.mean())
                print('l',l,'loss',loss.item())
            else:
                
               
                if(self.sampling_method=='rejectsampling'):
                    loss_RE = Z_sum.mean()
                    print('iteration',l,'average reject times:', averagetry/self.K,'REloss:',loss_RE.item(),end=' ')

                else:
                    #loss_RE = (pt.exp(Z_sum_forRE_untwisted2) * (Z_sum_forRE_untwisted1 )).mean() + pt.tensor(np.log(self.phi0_BPFL)).to(self.device)
                    #loss_RE = (pt.exp(Z_sum_forRE_untwisted2) * (Z_sum_forRE_untwisted1 )).mean() + pt.tensor(np.log(self.Z_true)).to(self.device)
                    loss_RE = (pt.exp(Z_sum_forRE_untwisted2) * (Z_sum_forRE_untwisted1 )).mean()
                    if(l % self.print_every == 0):
                        print('REloss:',loss_RE.item(),end=' ')

                #loss_CE = -(pt.exp(Z_sum_forCE2) * Z_sum_forCE1).mean()  + entropy_inCE 
                #loss_CE = -(pt.exp(Z_sum_forCE2 - pt.tensor(np.log(self.phi0_BPFL)).to(self.device)) * Z_sum_forCE1).mean()
                loss_CE = -(pt.exp(Z_sum_forCE2) * Z_sum_forCE1).mean()
                
                if(l % self.print_every == 0):
                    print('-CEloss:',loss_CE.item(),end=' ')

                loss_CERE = loss_RE + loss_CE
                if(l % self.print_every == 0):
                    print('CEREloss:',loss_CERE.item(),end = ' ')


                #early stop
                
                if(loss_RE <= self.standard):
                    break

                tt1 = time.time()
                if(self.train_goal == 'REtwisted'):
                    #train RE twisted
                    trueloss_RE = ((Z_sum.detach() + 1) * Z_sum_forloss).mean()    
                    self.zero_grad()
                    trueloss_RE.backward()
                    self.optimization_step()

                elif(self.train_goal == 'CEREtwisted'):
                    #trainCERE twisted
                    trueloss_CERE = ((Z_sum.detach() + 1) * Z_sum_forloss).mean() + loss_CE
                    self.zero_grad()
                    trueloss_CERE.backward()
                    self.optimization_step()

                elif(self.train_goal == 'CE'):
                    #train CE
                    trueloss_CE = loss_CE
                    self.zero_grad()
                    trueloss_CE.backward()
                    self.optimization_step()

                elif(self.train_goal == 'CERE'):
                    #train CERE untwisted
                    trueloss_CERE_untwisted = loss_CERE
                    self.zero_grad()
                    trueloss_CERE_untwisted.backward()
                    self.optimization_step()

                elif(self.train_goal == 'RE'):
                    #train RE untwisted
                    trueloss_RE_untwisted = loss_RE
                    self.zero_grad()
                    trueloss_RE_untwisted.backward()
                    self.optimization_step()

                else:
                    print('invalid loss choice!')


                tt2=time.time()
                if(l % self.print_every == 0):
                    print('training time:',tt2-tt1,end=' ')

            if self.log_gradient is True:
                grads = [params.grad for params in list(filter(lambda params: params.requires_grad, self.z_n.parameters()))
                        if params.grad is not None]
                grads_flat = pt.zeros(self.p)        
                i = 0
                for grad in grads:
                    grad_flat = grad.reshape(-1)
                    j = len(grad_flat)
                    grads_flat[i:i + j] = grad_flat
                    i += j

                self.gradient_log[l, :] = grads_flat.cpu().detach()

            self.loss_log_RE.append(loss_RE.item())
            self.loss_log_CERE.append(loss_CERE.item())
            self.loss_log_CE.append(loss_CE.item())
            #self.relvar_log.append(Rel_Var2.item())


            self.u_L2_loss.append(pt.mean(u_L2).item())
            if self.metastability_logs is not None:
                target, epsilon = self.metastability_logs
                self.particles_close_to_target.append(pt.mean((pt.sqrt(pt.sum((X - target)**2, 1)) <
                                                               epsilon).float()))


            t_1 = time.time()
            self.times.append(t_1 - t_0)
            if(l % self.print_every == 0):
                print('running time:',t_1 - t_0)


            
            
        if self.save_results is True:
            self.save_logs()
        print("training ended!")


        '''
        finally run a TPF to calculate Z with same particle numbers of BPF (K_BPF_small)
        1000 replciates
        '''

        ESS_ave = 0
        for ii in range(self.replicate_num):

            t_TPF1 = time.time()
              
            X_TPF = self.X_0.repeat(self.K_BPF_small, 1).to(self.device)
            

            y_observed_KBPFL = []#define y_OB for K_BPF_large particles
            for n in range(self.N + 1):
                y_observed_KBPFL.append(self.y_observed[n].repeat(self.K_BPF_small,1).to(self.device))
        
            g_prod = 1
            #g_prod = pt.ones(self.K_BPF_large).to(self.device)
            for n in range(self.N+1):
                g_k = pt.exp(-self.f(X_TPF , y_observed_KBPFL[n], n * self.delta_t))
                #first, define g^phi
                if(n == self.N):
                    gphi_k = g_k / pt.exp(self.Z_n_(X_TPF, n)).squeeze()
                    
                elif(n == 0):
                    #calculate tildephi
                    tildeN_TPF = 100
                    Z = self.Z_n_(X_TPF, n)
                    tildephi_TPF = pt.zeros_like(Z).to(self.device)     
                    for TPF_i in range(tildeN_TPF):
                            
                        Ui_TPF = self.sq_delta_t * pt.randn_like(X_TPF).to(self.device) + X_TPF + self.delta_t * self.b(X_TPF)

                        phii_TPF = pt.exp(self.Z_n_(Ui_TPF,n+1))
                        tildephi_TPF += phii_TPF
    
                    tildephi_TPF /= tildeN_TPF
                    

                    gphi_k = g_k * tildephi_TPF.squeeze()

                    

                else:
                    #calculate tildephi
                    tildeN_TPF = 100
                    Z = self.Z_n_(X_TPF, n)
                    tildephi_TPF = pt.zeros_like(Z).to(self.device)     
                    for TPF_i in range(tildeN_TPF):
                            
                        Ui_TPF = self.sq_delta_t * pt.randn_like(X_TPF).to(self.device) + X_TPF + self.delta_t * self.b(X_TPF)

                        phii_TPF = pt.exp(self.Z_n_(Ui_TPF,n+1))
                        tildephi_TPF += phii_TPF
    
                    tildephi_TPF /= tildeN_TPF

                    gphi_k = g_k * tildephi_TPF.squeeze() / pt.exp(self.Z_n_(X_TPF, n)).squeeze()
                    



                g_prod *= gphi_k.mean().item()
                #g_prod *= g_k
                    
                    
                #resample
                weights = gphi_k / pt.sum(gphi_k)
                indices = pt.multinomial(weights, self.K_BPF_small, replacement=True)
                ESS = 1.0 / np.sum((weights.detach().to('cpu').numpy())**2)
                ESS_ave += ESS
                
                X_TPF = X_TPF[indices]
                
                #print(X_TPF)
                #print(indices)

                #move particle position using rejecting sampling
                #reject sampling
                averagetry = 0
                for Xi in range(self.K_BPF_small):
                    ifrejectXi = 1
                    trynumber = 0
                    while(ifrejectXi == 1):
                        Xi_propose = pt.zeros_like(X_TPF[0].unsqueeze(0)).to(self.device)
                        Xi_propose = X_TPF[Xi].unsqueeze(0) + self.b(X_TPF[Xi].unsqueeze(0)) * self.delta_t + self.sq_delta_t * pt.randn_like(X_TPF[Xi].unsqueeze(0)).to(self.device)

                        Xi_acceptrate = pt.exp(self.Z_n_(Xi_propose,n+1)).squeeze().item()
                        # if(Xi == 0):
                        #     print(trynumber,'tiral,accept rate',Xi_acceptrate)
                        uniform_01 = np.random.uniform(0,1)
                        if(uniform_01 < Xi_acceptrate):
                            X_TPF[Xi] = Xi_propose.squeeze()#move forward
                            ifrejectXi = 0
                        else:
                            trynumber += 1
                            averagetry += 1
                if(n == 2):
                    print("when n = 2, reject sampling done with average reject:", averagetry / self.K_BPF_small, end ='  ')

            self.TPF_Z.append(g_prod)
            self.TPF_logZ.append(np.log(g_prod))
            t_TPF2 = time.time()
            print('1 time TPF ended in time', t_TPF2 - t_TPF1, 'Z = ', g_prod, 'log Z =', np.log(g_prod))
        
        
        ESS_ave /= (self.replicate_num * (self.N+1))
        print('ESS_ave = ', ESS_ave, 'ESS_ave percentage=', ESS_ave / self.K_BPF_small)


        



class Solver():

    def __init__(self, name, problem, lr=0.001, L=10000, K=50, delta_t=0.05,
                 approx_method='control', loss_method='log-variance', time_approx='outer',
                 learn_Y_0=False, adaptive_forward_process=True, detach_forward=False,
                 early_stopping_time=10000, random_X_0=False, compute_gradient_variance=0,
                 IS_variance_K=0, IS_variance_iter=1, metastability_logs=None, print_every=100, plot_trajectories=None,
                 seed=42, save_results=False, u_l2_error_flag=True, log_gradient=False, burgers_drift=False, verbose=True):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = problem.d
        self.T = problem.T
        self.X_0 = problem.X_0
        self.Y_0 = pt.tensor([0.0])
        self.X_u_opt = None

        # hyperparameters
        self.device = pt.device('cuda')
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(self.device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(self.device)
        self.N = int(np.floor(self.T / self.delta_t_np)) # number of steps
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size
        self.random_X_0 = random_X_0

        # learning properties
        self.loss_method = loss_method
        self.approx_method = approx_method
        self.learn_Y_0 = learn_Y_0
        self.adaptive_forward_process = adaptive_forward_process
        self.detach_forward = detach_forward
        self.early_stopping_time = early_stopping_time
        self.burgers_drift = burgers_drift

        self.has_ref_solution = hasattr(problem, 'u_true')
        self.u_l2_error_flag = u_l2_error_flag
        if self.has_ref_solution is False:
            self.u_l2_error_flag = False

        if self.loss_method == 'relative_entropy':
            self.adaptive_forward_process = True
        if self.loss_method == 'cross_entropy':
            self.learn_Y_0 = False

        # printing and logging
        self.print_every = print_every
        self.verbose = verbose
        self.verbose_NN = False
        self.save_results = save_results
        self.compute_gradient_variance = compute_gradient_variance
        self.IS_variance_K = IS_variance_K
        self.IS_variance_iter = IS_variance_iter
        self.metastability_logs = metastability_logs
        self.plot_trajectories = plot_trajectories
        self.metastability_logs = metastability_logs
        self.log_gradient = log_gradient
        self.print_gradient_norm = False

        # function approximation
        self.Phis = []
        self.time_approx = time_approx

        pt.manual_seed(seed)
        if self.approx_method == 'control':
            self.y_0 = SingleParam(lr=self.lr).to(self.device)
            if self.time_approx == 'outer':
                self.z_n = [DenseNet(d_in=self.d, d_out=self.d, lr=self.lr, seed=seed) for i in range(self.N)]
            elif self.time_approx == 'inner':
                #self.z_n = DenseNet(d_in=self.d + 1, d_out=self.d, lr=self.lr, seed=seed)
                self.z_n = MySequential(d_in=self.d + 1, d_out=self.d, lr=self.lr, seed=123)

        elif self.approx_method == 'value_function':
            if self.time_approx == 'outer':
                self.y_n = [DenseNet(d_in=self.d, d_out=1, lr=self.lr, seed=seed) for i in range(self.N)]
            elif self.time_approx == 'inner':
                self.y_n = [DenseNet(d_in=self.d + 1, d_out=1, lr=self.lr, seed=seed)]

        self.update_Phis()

        for phi in self.Phis:
            phi.train()

        if self.verbose_NN is True:
            if self.time_approx == 'outer':
                print('%d NNs, %d parameters in each network, total parameters = %d'
                      % (self.N, self.p, self.p * self.N))
            else:
                print('%d NNs, %d parameters in each network, total parameters = %d'
                      % (1, self.p, self.p))

        # logging
        self.Y_0_log = []
        self.loss_log = []
        self.u_L2_loss = []
        self.IS_rel_log = []
        self.times = []
        self.grads_rel_error_log = []
        self.particles_close_to_target = []

    def b(self, x):
        return self.problem.b(x)

    def sigma(self, x):
        return self.problem.sigma(x)

    def h(self, t, x, y, z):
        return self.problem.h(t, x, y, z)

    def f(self, x, t):
        return self.problem.f(x, t)

    def g(self, x):
        return self.problem.g(x)

    def u_true(self, x, t):
        return self.problem.u_true(x, t)

    def v_true(self, x, t):
        return self.problem.v_true(x, t)

    def update_Phis(self):
        if self.approx_method == 'control':
            if self.learn_Y_0 is True:
                if self.time_approx == 'outer':
                    self.Phis = self.z_n + [self.y_0]
                elif self.time_approx == 'inner':
                    self.Phis = [self.z_n, self.y_0]
            else:
                if self.time_approx == 'outer':
                    self.Phis = self.z_n
                elif self.time_approx == 'inner':
                    self.Phis = [self.z_n]
        elif self.approx_method == 'value_function':
            self.Phis = self.y_n
        for phi in self.Phis:
            phi.to(self.device)
        self.p = sum([np.prod(params.size()) for params in
          filter(lambda params: params.requires_grad,
                 self.Phis[0].parameters())])
        if self.log_gradient is True:
            self.gradient_log = pt.zeros(self.L, self.p)

    def loss_function(self, X, Y, Z_sum, l):
        if self.loss_method == 'moment':
            return (Y - self.g(X)).pow(2).mean()
        elif self.loss_method == 'log-variance':
            return (Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2)
        elif self.loss_method == 'log-variance-repa':
            return (l % 2 * 2 - 1) * ((Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2))
        elif self.loss_method == 'variance':
            return pt.var(pt.exp(- self.g(X) + Y))
        elif self.loss_method == 'log-variance_red':
            return ((-u_int - self.g(X)).pow(2).mean() - 2 * ((-u_int - self.g(X)) * u_W_int).mean()
                    + 2 * u_int.mean() - (-u_int - self.g(X)).mean().pow(2))
        elif self.loss_method == 'log-variance_red_2':
            return ((-u_int - self.g(X)).pow(2).mean() + 2 * (self.g(X) * u_W_int).mean()
                    - double_int.mean() + 2 * u_int.mean() - (-u_int - self.g(X)).mean().pow(2))
        elif self.loss_method == 'relative_entropy':
            return (Z_sum + self.g(X)).mean()
        elif self.loss_method == 'relative_entropy_BSDE':
            return (Z_sum + self.g(X)).mean()
        elif self.loss_method == 'cross_entropy':
            if self.adaptive_forward_process is True:
                return (Y * pt.exp(-self.g(X) + Y.detach())).mean()
            return (Y * pt.exp(-self.g(X))).mean()
        elif self.loss_method == 'relative_entropy_log-variance':
            if l < 1000:
                return ((Z_sum + self.g(X))).mean()
            return (Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2)
        elif self.loss_method == 'reparametrization':
            return (Z_sum + self.g(X)).mean()

    def zero_grad(self):
        for phi in self.Phis:
            phi.optim.zero_grad()

    def optimization_step(self):
        for phi in self.Phis:
            phi.optim.step()

    def gradient_descent(self, X, Y, Z_sum, l, additional_loss):
        self.zero_grad()

        if self.loss_method == 'log-variance-y_0':

            loss_1 = pt.var(Y - self.g(X))
            loss_1.backward(retain_graph=True)
            self.z_n.optim.step()

            if self.learn_Y_0 is True:
                loss_2 = pt.mean(Y - self.g(X))**2
                loss_2.backward(retain_graph=True)
                self.y_0.optim.step()
            else:
                loss_2 = 0

            return loss_1 + loss_2

        loss = self.loss_function(X, Y, Z_sum, l) + additional_loss
        loss.backward()
        self.optimization_step()
        return loss

    def flatten_gradient(self, k, grads, grads_flat):
        i = 0
        for grad in grads:
            grad_flat = grad.reshape(-1)
            j = len(grad_flat)
            grads_flat[k, i:i + j] = grad_flat
            i += j
        return grads_flat

    def get_gradient_variances(self, X, Y):
        grads_mean = pt.zeros(self.N, self.p)
        grads_var = pt.zeros(self.N, self.p)

        for n in range(self.N):

            grads_Y_flat = pt.zeros(self.K, self.p)

            for k in range(self.K):
                self.zero_grad()
                Y[k].backward(retain_graph=True)

                grad_Y = [params.grad for params in list(filter(lambda params:
                                                                params.requires_grad,
                                                                self.z_n[n].parameters()))
                          if params.grad is not None]

                grads_Y_flat = self.flatten_gradient(k, grad_Y, grads_Y_flat)

            grads_g_X_flat = pt.zeros(self.K, self.p)

            if self.adaptive_forward_process is True:

                for k in range(self.K):
                    self.zero_grad()
                    self.g(X[0, :].unsqueeze(0)).backward(retain_graph=True)

                    grad_g_X = [params.grad for params in list(filter(lambda params:
                                                                      params.requires_grad,
                                                                      self.z_n[n].parameters()))
                                if params.grad is not None]

                    grads_g_X_flat = self.flatten_gradient(k, grad_g_X, grads_g_X_flat)

            if self.loss_method == 'moment':
                grads_flat = 2 * (Y - self.g(X)).unsqueeze(1) * (grads_Y_flat - grads_g_X_flat)
            elif self.loss_method == 'log-variance':
                grads_flat = 2 * (((Y - self.g(X)).unsqueeze(1)
                                   - pt.mean((Y - self.g(X)).unsqueeze(1), 0).unsqueeze(0))
                                  * (grads_Y_flat - grads_g_X_flat
                                     - pt.mean(grads_Y_flat - grads_g_X_flat, 0).unsqueeze(0)))

            grads_mean[n, :] = pt.mean(grads_flat, dim=0)
            grads_var[n, :] = pt.var(grads_flat, dim=0)

        grads_rel_error = pt.sqrt(grads_var) / grads_mean
        grads_rel_error[grads_rel_error != grads_rel_error] = 0
        return grads_rel_error

    def state_dict_to_list(self, sd):
        sd_list = {}
        for name in sd:
            if type(sd[name]) == pt.Tensor:
                sd_list[name] = sd[name].detach().cpu().numpy().tolist()
            else:
                sd_list[name] = sd[name]
        return sd_list

    def list_to_state_dict(self, l):
         return {param: pt.tensor(l[param]) if type(l[param]) == list else l[param] for param in l}

    def save_logs(self, model_name='model'):
        # currently does not work for all modi
        logs = {'name': self.name, 'date': self.date, 'd': self.d, 'T': self.T,
                'seed': self.seed, 'delta_t': self.delta_t_np, 'N': self.N, 'lr': self.lr,
                'K': self.K, 'loss_method': self.loss_method, 'learn_Y_0': self.learn_Y_0,
                'adaptive_forward_process': self.adaptive_forward_process,
                'Y_0_log': self.Y_0_log, 'loss_log': self.loss_log, 'u_L2_loss': self.u_L2_loss,
                'Phis_state_dict': [self.state_dict_to_list(z.cpu().state_dict()) for z in self.Phis]}

        path_name = 'logs/%s_%s_%s.json' % (model_name, self.name, self.date)
        i = 1
        while os.path.isfile(path_name):
            i += 1
            path_name = 'logs/%s_%s_%s_%d.json' % (model_name, self.name, self.date, i)

        with open(path_name, 'w') as f:
            json.dump(logs, f, indent=2)

    def save_networks(self):
        data_dict = {}
        idx = 0
        for z in self.Phis:
            key = 'nn%d' % idx
            data_dict[key] = z.state_dict()
            idx += 1
        path_name = 'output/%s_%s.pt' % (self.name, self.date)
        pt.save(data_dict, path_name)
        print('\nnetworks data has been stored to file: %s' % path_name)

    def load_networks(self, cp_name):
        print('\nload network data from file: %s' % cp_name)
        checkpoint = pt.load(cp_name)
        idx = 0
        for z in self.Phis:
            key = 'nn%d' % idx
            z.load_state_dict(checkpoint[key])
            z.eval()
            idx += 1

    def compute_grad_Y(self, X, n):
        Y_n_eval = self.Y_n(X, n).squeeze(1).sum()
        Y_n_eval.backward(retain_graph=True)
        Z, = pt.autograd.grad(Y_n_eval, X, create_graph=True)
        Z = pt.mm(self.sigma(X), Z.t()).t()
        return Z

    def Y_n(self, X, t):
        n = int(np.ceil(t / self.delta_t))
        if self.time_approx == 'outer':
            return self.y_n[n](X)
        elif self.time_approx == 'inner':
            t_X = pt.cat([pt.ones([X.shape[0], 1]) * t, X], 1)
            return self.y_n[0](t_X)

    def Z_n_(self, X, n):
        if self.approx_method == 'control':
            if self.time_approx == 'outer':
                n = max(0, min(n, self.N - 1))
                return self.z_n[n](X)
            elif self.time_approx == 'inner':
                t_X = pt.cat([pt.ones([X.shape[0], 1]).to(self.device) * n * self.delta_t, X], 1)
                return self.z_n(t_X)
        if self.approx_method == 'value_function':
            return self.compute_grad_Y(X, n)

    def Z_n(self, X, t):
        n = int(pt.ceil(t / self.delta_t))
        return self.Z_n_(X, n)

    def initialize_training_data(self):
        X = self.X_0.repeat(self.K, 1).to(self.device)
        if self.random_X_0 is True:
            X = pt.randn(self.K, self.d).to(self.device)
        Y = self.Y_0.repeat(self.K).to(self.device)
        if self.approx_method == 'value_function':
            X = pt.autograd.Variable(X, requires_grad=True)
            Y = self.Y_n(X, 0)[:, 0]
        elif self.learn_Y_0 is True:
            Y = self.y_0(X)
            self.Y_0_log.append(Y[0].item())
        Z_sum = pt.zeros(self.K).to(self.device)
        u_L2 = pt.zeros(self.K).to(self.device)
        u_int = pt.zeros(self.K).to(self.device)
        u_W_int = pt.zeros(self.K).to(self.device)
        double_int = pt.zeros(self.K).to(self.device)

        xi = pt.randn(self.K, self.d, self.N + 1).to(self.device)
        return X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi

    def train_LSE_with_reference(self):
        if self.approx_method != 'control':
            print('only learn control with reference solution!')
        if self.has_ref_solution == False:
            print('reference solution is needed!')

        print('\nd = %d, L = %d, K = %d, delta_t = %.2e, N = %d, lr = %.2e, %s, %s, %s, %s\n' % (self.d, self.L, self.K, self.delta_t_np, self.N, self.lr, self.approx_method, self.time_approx, self.loss_method, 'adaptive' if self.adaptive_forward_process else ''))

        xb = 2.0
        X = pt.linspace(-xb, xb, 200).unsqueeze(1)

        for l in range(self.L):
            t_0 = time.time()
            loss = 0.0
            for n in range(self.N):
                loss += pt.sum((- self.Z_n_(X, n) - pt.tensor(self.u_true(X, n * self.delta_t_np)).float())**2) * self.delta_t
            self.zero_grad()
            loss.backward()
            self.optimization_step()

            self.loss_log.append(loss.item())

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                string = ('%d - loss: %.3e - time/iter: %.2fs' % (l, self.loss_log[-1],
                          np.mean(self.times[-self.print_every:])))
                print(string + ' - gradient l_inf: %.3e' %
                      (np.array([max([pt.norm(params.grad.data, float('inf')).item() for params
                                      in filter(lambda params: params.requires_grad,
                                                phi.parameters())])
                                 for phi in self.Phis]).max()))

        #   self.save_networks()

    def train(self):

        pt.manual_seed(self.seed)

        if self.verbose is True:
            print('d = %d, L = %d, K = %d, delta_t = %.2e, lr = %.2e, %s, %s, %s, %s'
                  % (self.d, self.L, self.K, self.delta_t_np, self.lr, self.approx_method,
                     self.time_approx, self.loss_method,
                     'adaptive' if self.adaptive_forward_process else ''))

        for l in range(self.L):
            t_0 = time.time()

            X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi = self.initialize_training_data()
            additional_loss = pt.zeros(self.K)


            if self.loss_method == 'reparametrization':
                z_n_copy = deepcopy(self.z_n)

            for n in range(self.N):
                if self.approx_method == 'value_function':
                    if n > 0:
                        additional_loss += (self.Y_n(X, n)[:, 0] - Y).pow(2)
                if self.loss_method == 'log-variance-repa' and l % 2 == 0:
                    z_n_copy = deepcopy(self.z_n)
                    t_X = pt.cat([pt.ones([X.shape[0], 1]).to(self.device) * n * self.delta_t, X], 1)
                    Z = z_n_copy(t_X)
                else:
                    Z = self.Z_n_(X, n)

                c = pt.zeros(self.d, self.K).to(self.device)
                if self.adaptive_forward_process is True:
                    if self.burgers_drift is True:
                        c = pt.ones(self.d, self.K).to(self.device) * (Y.unsqueeze(0) - (2 + self.d) / (2 * self.d))
                    else:
                        c = -self.Z_n_(X, n).t()
                        #c = -Z.t()

                if self.loss_method == 'reparametrization':
                    if self.time_approx == 'outer':
                        n = max(0, min(n, self.N - 1))
                        z_n_copy = deepcopy(self.z_n[n])
                        v = -z_n_copy(X)
                    else:
                        t_X = pt.cat([pt.ones([X.shape[0], 1]).to(self.device) * n * self.delta_t, X], 1)
                        v = -z_n_copy(t_X)

                if self.detach_forward is True or (self.loss_method == 'log-variance-repa' and l % 2 == 1):
                    c = c.detach()

                X = (X + (self.b(X) + pt.mm(self.sigma(X), c).t()) * self.delta_t
                     + pt.mm(self.sigma(X), xi[:, :, n + 1].t()).t() * self.sq_delta_t)

                #X = (X + (self.b(X) + pt.bmm(self.sigma(X), c.t().unsqueeze(2)).squeeze(2)) * self.delta_t 
                #     + pt.bmm(self.sigma(X), xi[:, :, n + 1].unsqueeze(2)).squeeze(2) * self.sq_delta_t)

                Y = (Y + (-self.h(self.delta_t * n, X, Y, Z) + pt.sum(Z * c.t(), 1)) * self.delta_t
                     + pt.sum(Z * xi[:, :, n + 1], 1) * self.sq_delta_t)

                if self.loss_method == 'reparametrization':
                    Z_sum += (-0.5 * pt.sum(v**2, 1) * self.delta_t + pt.sum(v * c.t(), 1) * self.delta_t
                              + pt.sum(v * xi[:, :, n + 1], 1) * self.sq_delta_t)

                if 'relative_entropy' in self.loss_method:
                    #Z_sum += 0.5 * pt.sum((-Z)**2, 1) * self.delta_t
                    Z_sum += (0.5 * pt.sum(Z**2, dim=1) + self.f(X, n * self.delta_t)) * self.delta_t
                    #Z_sum += self.h(n * self.delta_t, X, Y, Z) * self.delta_t
                    if self.loss_method == 'relative_entropy_BSDE':
                        Z_sum += pt.sum(-Z * xi[:, :, n + 1], 1) * self.sq_delta_t

                if self.u_l2_error_flag is True:
                    u_L2 += pt.sum((-Z
                                    - pt.tensor(self.u_true(X.cpu().detach(), n * self.delta_t_np)).t().float().to(self.device))**2
                                   * self.delta_t, 1)

            if self.compute_gradient_variance > 0 and l % self.compute_gradient_variance == 0:
                self.grads_rel_error_log.append(pt.mean(self.get_gradient_variances(X, Y)).item())

            loss = self.gradient_descent(X, Y, Z_sum, l, additional_loss.mean())

            if self.log_gradient is True:
                grads = [params.grad for params in list(filter(lambda params: params.requires_grad, self.z_n.parameters()))
                        if params.grad is not None]
                grads_flat = pt.zeros(self.p)        
                i = 0
                for grad in grads:
                    grad_flat = grad.reshape(-1)
                    j = len(grad_flat)
                    grads_flat[i:i + j] = grad_flat
                    i += j

                self.gradient_log[l, :] = grads_flat.cpu().detach()

            self.loss_log.append(loss.item())
            self.u_L2_loss.append(pt.mean(u_L2).item())
            if self.metastability_logs is not None:
                target, epsilon = self.metastability_logs
                self.particles_close_to_target.append(pt.mean((pt.sqrt(pt.sum((X - target)**2, 1)) <
                                                               epsilon).float()))

            if self.IS_variance_K > 0 and l % self.IS_variance_iter == 0:
                _, _, rel_IS = do_importance_sampling_me(self.problem, self, self.IS_variance_K)
                #_, _, rel_naive, _, _, rel_IS = do_importance_sampling(self.problem, self,
                #                                                                 self.IS_variance_K,
                #                                                                 control='approx',
                #                                                                 verbose=False,
                #                                                                 plot_trajectories=self.plot_trajectories)
                self.IS_rel_log.append(rel_IS)

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if self.verbose is True:
                if l % self.print_every == 0:
                    string = ('%d - loss: %.4e - u L2: %.4e - time/iter: %.2fs'
                              % (l, self.loss_log[-1], self.u_L2_loss[-1],
                                 np.mean(self.times[-self.print_every:])))
                    if self.learn_Y_0 is True:
                            string += ' - Y_0: %.4e' % self.Y_0_log[-1]
                    if self.IS_variance_K > 0:
                        string += ' - rel IS: %.3e' % rel_IS
                    print(string)
                    if self.print_gradient_norm is True:
                        print('gradient l_inf: %.3e' %
                              (np.array([max([pt.norm(params.grad.data, float('inf')).item() for params in
                                              filter(lambda params: hasattr(params.grad, 'data'),
                                                     phi.parameters())])
                                         for phi in self.Phis]).max()))

            if self.early_stopping_time is not None:
                if ((l > self.early_stopping_time) and
                        (np.std(self.u_L2_loss[-self.early_stopping_time:])
                         / self.u_L2_loss[-1] < 0.02)):
                    break

        if self.save_results is True:
            self.save_logs()


class EllipticSolver():

    def __init__(self, problem, name, seed=42, delta_t=0.01, N=50, lr=0.001, L=100000, K=200, K_boundary=50,
                 alpha=[1.0, 1.0], adaptive_forward_process=False, detach_forward=True, print_every=100, verbose=True, 
                 approx_method='Y', sample_center=False, loss_method='diffusion', loss_with_stopped=False, K_test_log=None,
                 PINN_log_variance=False, log_loss_parts=False, boundary_loss=True, boundary_type='Dirichlet',
                 variance_moment_split=False, full_hessian=False, uniform_square=False):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = problem.d

        # hyperparameters
        self.device = pt.device('cuda')
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(self.device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(self.device)
        self.N = N # trajectory length
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size in domain
        self.K_original = K # batch size in domain
        self.K_boundary = K_boundary # batch size on boundary
        self.alpha = alpha # weights
        self.boundary_type = boundary_type

        # learning properties
        self.adaptive_forward_process = adaptive_forward_process
        self.detach_forward = detach_forward
        self.approx_method = approx_method
        self.sample_center = sample_center
        self.loss_method = loss_method
        self.loss_with_stopped = loss_with_stopped
        self.boundary_loss = boundary_loss
        self.PINN_log_variance = PINN_log_variance
        self.variance_moment_split = variance_moment_split
        self.full_hessian = full_hessian
        self.uniform_square = uniform_square

        # printing and logging
        self.print_every = print_every
        self.verbose = verbose

        # function approximation
        pt.manual_seed(seed)
        if self.approx_method == 'Y':
            self.V = DenseNet(d_in=self.d, d_out=1, lr=self.lr, seed=seed).to(self.device)
        elif self.approx_method == 'Z':
            self.y_0 = SingleParam(lr=self.lr).to(self.device)
            self.Z = DenseNet(d_in=self.d, d_out=self.d, lr=self.lr, seed=seed).to(self.device)

        # logging
        self.K_test_log = K_test_log
        self.Y_0_log = []
        self.loss_log = []
        self.loss_log_domain = []
        self.loss_log_boundary = []
        self.u_L2_log = []
        self.V_L2_log = []
        self.V_test_L2 = []
        self.V_test_abs = []
        self.V_test_rel_abs = []
        self.times = []
        self.lambda_log = []
        self.log_loss_parts = log_loss_parts
        self.K_log = []

    def train(self):

        pt.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.loss_method == 'PINN':
            self.train_PINN()
            return None

        for l in range(self.L):

            t_0 = time.time()

            loss = 0

            if self.sample_center:
                X_center = pt.zeros(1, 1)
                loss += pt.mean((self.V(X_center).squeeze() - self.problem.v_true(X_center).squeeze())**2)
            # sample uniformly on boundary
            if self.problem.boundary == 'sphere':
                X_boundary = pt.randn(self.K_boundary, self.d).to(self.device)
                X_boundary = self.problem.boundary_distance * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1)
            elif self.problem.boundary == 'two_spheres':
                X_boundary = pt.randn(self.K_boundary, self.problem.d).to(self.device)
                X_boundary = (pt.tensor([self.problem.boundary_distance_1] * int(self.K_boundary / 2) + 
                                        [self.problem.boundary_distance_2] * int(self.K_boundary / 2)).unsqueeze(1).to(self.device)
                              * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1))
            elif self.problem.boundary == 'square':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_l
                X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_l
                X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_r
                if self.problem.one_boundary:
                    X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_r
                    X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_r
            elif self.problem.boundary == 'square-corner':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_corner) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_corner
                X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_corner
                X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_corner


            if self.loss_method not in ['BSDE-4', 'BSDE'] and self.boundary_loss:
                if self.boundary_type == 'Dirichlet':
                    loss += self.alpha[1] * pt.mean((self.V(X_boundary).squeeze() - self.problem.g(X_boundary))**2)
                elif self.boundary_type == 'Neumann':
                    X_boundary = pt.autograd.Variable(X_boundary, requires_grad=True)
                    Y_ = self.V(X_boundary)
                    Y_eval = Y_.squeeze().sum()
                    Y_eval.backward(retain_graph=True)
                    grad_V, = pt.autograd.grad(Y_eval, X_boundary, create_graph=True)
                    loss += self.alpha[1] * pt.mean((pt.sum(grad_V * X_boundary, 1) - pt.sum(self.problem.g(X_boundary) * X_boundary, 1))**2)

            if self.problem.boundary == 'sphere':
                if self.uniform_square:
                    X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                    X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)).to(self.device)
                else:
                    X = pt.randn(self.K, self.problem.d).to(self.device)
                    X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)**(1 / self.problem.d)).to(self.device)
            elif self.problem.boundary == 'two_spheres':
                if self.uniform_square:
                    X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                    X = X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K, self.problem.d).to(self.device) * (self.problem.boundary_distance_2 - self.problem.boundary_distance_1) + self.problem.boundary_distance_1)
                else:
                    X = pt.randn(self.K_original, self.problem.d).to(self.device)
                    X = self.problem.boundary_distance_2 * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K_original).unsqueeze(1)**(1 / self.problem.d)).to(self.device)
                    selection = pt.sqrt(pt.sum(X**2, 1)) > self.problem.boundary_distance_1
                    X = X[selection, :]
                    self.K = int(pt.sum(selection))
            elif self.problem.boundary == 'square':
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l
            elif self.problem.boundary == 'square-corner':
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l
                X[pt.all(X > self.problem.X_corner, 1), :] = -X[pt.all(X > self.problem.X_corner, 1), :]

            X = pt.autograd.Variable(X, requires_grad=True)
            Y = pt.zeros(self.K).to(self.device)
            if self.loss_method in ['BSDE-2', 'BSDE-4', 'BSDE', 'diffusion']:
                Y = self.V(X).squeeze()

            #lambda_log.append(lambda_(X)[0].item())
            stopped = pt.zeros(self.K).bool().to(self.device)
            hitting_times = pt.zeros(self.K)
            V_L2 = pt.zeros(self.K)
            K_count = 0

            #phi_0 = self.V(X).squeeze()

            for n in range(self.N):

                Y_ = self.V(X)
                Y_eval = Y_.squeeze().sum()
                Y_eval.backward(retain_graph=True)
                Z, = pt.autograd.grad(Y_eval, X, create_graph=True)
                Z = pt.mm(self.problem.sigma(X).t(), Z.t()).t()

                xi = pt.randn(self.K, self.d).to(self.device)

                selection = ~stopped
                K_selection = pt.sum(selection)
                if K_selection == 0:
                    break

                V_L2[selection] += ((self.V(X[selection]).squeeze() - pt.tensor(self.problem.v_true(X[selection].detach())).float().squeeze())**2).detach().cpu() * self.delta_t_np

                c = pt.zeros(self.d, self.K).to(self.device)
                if self.adaptive_forward_process is True:
                    c = -Z.t()
                if self.detach_forward is True:
                    c = c.detach()

                X_proposal = (X + ((self.problem.b(X) + pt.mm(self.problem.sigma(X), c).t()) * self.delta_t
                     + pt.mm(self.problem.sigma(X), xi.t()).t() * self.sq_delta_t) * selection.float().unsqueeze(1).repeat(1, self.d))

                hitting_times[selection] += 1
                if self.problem.boundary == 'sphere':
                    new_selection = pt.all(pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) < self.problem.boundary_distance, 1).to(self.device)
                elif self.problem.boundary == 'two_spheres':
                    new_selection = ((pt.sqrt(pt.sum(X**2, 1)) > self.problem.boundary_distance_1) & (pt.sqrt(pt.sum(X**2, 1)) < self.problem.boundary_distance_2)).to(self.device)
                elif self.problem.boundary == 'square':
                    if self.problem.one_boundary:
                        new_selection = pt.all((X_proposal <= self.problem.X_r), 1).to(self.device)
                    else:
                        new_selection = pt.all((X_proposal >= self.problem.X_l) & (X_proposal <= self.problem.X_r), 1).to(self.device)
                elif self.problem.boundary == 'square-corner':
                    new_selection = pt.any((X_proposal <= self.problem.X_r), 1).to(self.device)

                if self.loss_method == 'BSDE-2':
                    loss += self.alpha[0] * pt.mean((Y_.squeeze() - Y)**2 * (new_selection & ~stopped).float())

                if self.loss_method in ['BSDE-2', 'BSDE-4']:
                    Y = (Y + ((- self.problem.h(X, Y, Z) #- lambda_(X) * Y_.squeeze() #  lambda_(X) 
                               + pt.sum(Z * c.t(), 1)) * self.delta_t + pt.sum(Z * xi, 1) * self.sq_delta_t) * (new_selection & ~stopped).float())
                else:
                    Y = (Y + ((- self.problem.h(X, Y_.squeeze(), Z) #- lambda_(X) * Y_.squeeze() #  lambda_(X) 
                               + pt.sum(Z * c.t(), 1)) * self.delta_t + pt.sum(Z * xi, 1) * self.sq_delta_t) * (new_selection & ~stopped).float())

                X_ = X
                X = (X * (~new_selection | stopped).float().unsqueeze(1).repeat(1, self.d)
                     + X_proposal * (new_selection & ~stopped).float().unsqueeze(1).repeat(1, self.d))

                if self.loss_method in ['BSDE', 'diffusion']:
                    K_count += pt.sum(new_selection & ~stopped)

                if pt.sum(~new_selection & ~stopped) > 0:
                    stopped[~new_selection & ~stopped] += True

                if self.loss_method == 'BSDE-3':
                    loss += self.alpha[0] * pt.mean((self.V(X).squeeze() - Y_.squeeze() + (self.problem.h(X_, Y_.squeeze(), Z)
                                                                 - pt.sum(Z * c.t(), 1)) * self.delta_t 
                                     - pt.sum(Z * xi, 1) * self.sq_delta_t)**2 * (new_selection & ~stopped).float())

            if self.loss_method == 'diffusion':
                if self.variance_moment_split:
                    loss += self.alpha[0] * (pt.var(self.V(X).squeeze() - Y) + pt.mean((self.V(X[:1, :]).squeeze() - Y[:1])**2))
                    #loss += self.alpha[0] * (pt.var(self.V(X).squeeze() - Y) + pt.mean(self.V(X[:10, :]).squeeze() - Y[:10])**2)
                else:
                    loss += self.alpha[0] * pt.mean((self.V(X).squeeze() - Y)**2)
            #lambda_.zero_grad()

            if self.loss_method in ['BSDE', 'diffusion']:
                self.K_log.append(K_count.item())

            if self.loss_method in ['BSDE-4', 'BSDE']:
                if pt.sum(stopped).item() != self.K:
                    print('Not all trajectories stopped.')
                loss += pt.mean((self.problem.g(X) - Y)**2)

            if self.loss_with_stopped:
                loss += pt.mean((self.problem.g(X[stopped, :]) - Y[stopped])**2)

            self.V.zero_grad()
            loss.backward(retain_graph=True)
            self.V.optim.step()

            #lambda_.optim.step()

            self.loss_log.append(loss.item())
            self.V_L2_log.append(pt.mean(V_L2).item())
            if self.K_test_log is not None:
                L2_error, mean_abolute_error, mean_relative_error = compute_test_error(self, self.problem, self.K_test_log, self.device)
                self.V_test_L2.append(L2_error)
                self.V_test_abs.append(mean_abolute_error)
                self.V_test_rel_abs.append(mean_relative_error)

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if self.verbose:
                if l % self.print_every == 0:
                    print('%d - loss = %.4e, v L2 error = %.4e, n = %d, active: %d/%d, %.2f' % 
                          (l, self.loss_log[-1], self.V_L2_log[-1], n, K_selection, self.K, np.mean(self.times[-self.print_every:])))

    def train_PINN(self):
        for l in range(self.L):

            t_0 = time.time()

            if self.problem.boundary == 'sphere':
                X_boundary = pt.randn(self.K_boundary, self.d).to(self.device)
                X_boundary = self.problem.boundary_distance * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1)
            elif self.problem.boundary == 'two_spheres':
                X_boundary = pt.randn(self.K_boundary, self.problem.d).to(self.device)
                X_boundary = (pt.tensor([self.problem.boundary_distance_1] * int(self.K_boundary / 2) + 
                                        [self.problem.boundary_distance_2] * int(self.K_boundary / 2)).unsqueeze(1).to(self.device)
                              * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1))
            elif self.problem.boundary == 'square':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_l
                X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_l
                X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_r
                if self.problem.one_boundary:
                    X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_r
                    X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_r
            elif self.problem.boundary == 'square-corner':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_corner) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_corner
                X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_corner
                X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_corner

            if self.problem.boundary == 'sphere':
                if self.uniform_square:
                    X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                    X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)).to(self.device)
                else:
                    X = pt.randn(self.K, self.problem.d).to(self.device)
                    X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)**(1 / self.problem.d)).to(self.device)
            elif self.problem.boundary == 'two_spheres':
                if self.uniform_square:
                    X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                    X = X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K, self.problem.d).to(self.device) * (self.problem.boundary_distance_2 - self.problem.boundary_distance_1) + self.problem.boundary_distance_1)
                else:
                    X = pt.randn(self.K_original, self.problem.d).to(self.device)
                    X = self.problem.boundary_distance_2 * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K_original).unsqueeze(1)**(1 / self.problem.d)).to(self.device)
                    selection = pt.sqrt(pt.sum(X**2, 1)) > self.problem.boundary_distance_1
                    X = X[selection, :]
                    self.K = int(pt.sum(selection))
            elif self.problem.boundary == 'square':
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l
            elif self.problem.boundary == 'square-corner':
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l
                X[pt.all(X > self.problem.X_corner, 1), :] = -X[pt.all(X > self.problem.X_corner, 1), :]

            second_derivatives = pt.zeros(X.shape[0]).to(self.device)

            X = pt.autograd.Variable(X, requires_grad=True)

            V_eval = self.V(X).squeeze()
            grad = pt.autograd.grad(V_eval, X, grad_outputs=pt.ones(self.K).to(self.device), create_graph=True)[0]

            if self.full_hessian:
                for i, x in enumerate(X):
                    hess = pt.autograd.functional.hessian(self.V, x.unsqueeze(0), create_graph=True).squeeze()
                    second_derivatives[i] = pt.sum(pt.diagonal(pt.mm(pt.mm(self.problem.B, self.problem.B.t()), hess)))
            else:
                for k in range(grad.shape[1]):
                    rad_grad_u_xx = pt.autograd.grad(grad[:, k], X, grad_outputs=pt.ones(self.K).to(self.device), create_graph=True)[0][:, k]
                    second_derivatives += rad_grad_u_xx
                second_derivatives = self.problem.B[0, 0]**2 * second_derivatives

            if self.PINN_log_variance:
                loss = self.alpha[0] * pt.var(0.5 * second_derivatives + pt.sum(self.problem.b(X) * grad, 1) 
                                              + self.problem.h(X, self.V(X).squeeze(), pt.mm(self.problem.B, grad.t()).t()))
            else:
                loss = self.alpha[0] * pt.mean((0.5 * second_derivatives + pt.sum(self.problem.b(X) * grad, 1) 
                                                + self.problem.h(X, self.V(X).squeeze(), pt.mm(self.problem.B, grad.t()).t()))**2) 
            if self.log_loss_parts:
                self.loss_log_domain.append(loss.item() / self.alpha[0])
            if self.boundary_loss:
                loss += self.alpha[1] * pt.mean((self.V(X_boundary).squeeze() - self.problem.g(X_boundary))**2)
            if self.log_loss_parts:
                self.loss_log_boundary.append(pt.mean((self.V(X_boundary).squeeze() - self.problem.g(X_boundary))**2).item())

            self.V.zero_grad()

            loss.backward(retain_graph=True)
            self.V.optim.step()

            self.V_L2_log.append(pt.mean(((self.V(X).squeeze() - pt.tensor(self.problem.v_true(X.detach())).float().squeeze())**2).detach().cpu() * self.delta_t_np).item())
            self.loss_log.append(loss.item())
            if self.K_test_log is not None:
                L2_error, mean_abolute_error, mean_relative_error = compute_test_error(self, self.problem, self.K_test_log, self.device)
                self.V_test_L2.append(L2_error)
                self.V_test_abs.append(mean_abolute_error)
                self.V_test_rel_abs.append(mean_relative_error)

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                print('%d - loss = %.4e - v L2 error = %.4e - %.2f' % (l, self.loss_log[-1], self.V_L2_log[-1], np.mean(self.times[-self.print_every:])))


class GeneralSolver():
    
    def __init__(self, problem, name, seed=42, delta_t=0.01, N=50, lr=0.001, L=100000, K=200, K_boundary=50,
                 alpha=[1.0, 1.0, 1.0], adaptive_forward_process=False, detach_forward=True, print_every=100, verbose=True, 
                 approx_method='Y', sample_center=False, loss_method='diffusion', loss_with_stopped=False, K_test_log=None,
                 PINN_log_variance=False, log_loss_parts=False, boundary_loss=True, full_hessian=False, uniform_square=False,
                 solve_linear_L2_projection=False):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = problem.d

        # hyperparameters
        self.device = pt.device('cuda')
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(self.device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(self.device)
        self.N = N # trajectory length
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size in domain
        self.K_original = K
        self.K_boundary = K_boundary # batch size on boundary
        self.alpha = alpha # weights

        # learning properties
        self.adaptive_forward_process = adaptive_forward_process
        self.detach_forward = detach_forward
        self.approx_method = approx_method
        self.sample_center = sample_center
        self.loss_method = loss_method
        self.loss_with_stopped = loss_with_stopped
        self.boundary_loss = boundary_loss
        self.PINN_log_variance = PINN_log_variance
        self.full_hessian = full_hessian
        self.uniform_square = uniform_square
        self.solve_linear_L2_projection = solve_linear_L2_projection

        # printing and logging
        self.print_every = print_every
        self.verbose = verbose

        # function approximation
        pt.manual_seed(seed)
        if self.approx_method == 'Y':
            self.V = DenseNet(d_in=self.d + 1, d_out=1, lr=self.lr, seed=seed).to(self.device)
        elif self.approx_method == 'Z':
            self.y_0 = SingleParam(lr=self.lr).to(self.device)
            self.Z = DenseNet(d_in=self.d + 1, d_out=self.d, lr=self.lr, seed=seed).to(self.device)

        # logging
        self.K_test_log = K_test_log
        self.Y_0_log = []
        self.loss_log = []
        self.loss_log_domain = []
        self.loss_log_boundary = []
        self.u_L2_log = []
        self.V_L2_log = []
        self.V_test_L2 = []
        self.V_test_abs = []
        self.V_test_rel_abs = []
        self.times = []
        self.lambda_log = []
        self.log_loss_parts = log_loss_parts
        self.K_log = []

    def train(self):

        pt.manual_seed(self.seed)

        if self.loss_method == 'PINN':
            self.train_PINN()
            return None

        for l in range(self.L):

            t_0 = time.time()

            loss = 0

            if self.sample_center:
                X_center = pt.zeros(1, 1)
                loss += pt.mean((self.V(X_center).squeeze() - self.problem.v_true(X_center).squeeze())**2)

            # sample uniformly on boundary
            if self.problem.boundary == 'sphere':
                X_boundary = pt.randn(self.K_boundary, self.d).to(self.device)
                X_boundary = self.problem.boundary_distance * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1)
            elif self.problem.boundary == 'two_spheres':
                X_boundary = pt.randn(self.K_boundary, self.problem.d).to(self.device)
                X_boundary = (pt.tensor([self.problem.boundary_distance_1] * int(self.K_boundary / 2) + 
                                        [self.problem.boundary_distance_2] * int(self.K_boundary / 2)).unsqueeze(1).to(self.device)
                              * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1))
            elif self.problem.boundary == 'square':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_l
                X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_l
                X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_r
                if self.problem.one_boundary:
                    X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_r
                    X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_r

            if self.problem.boundary in ['sphere', 'unbounded']:
                if self.uniform_square:
                    X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                    X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)).to(self.device)
                else:
                    X = pt.randn(self.K, self.problem.d).to(self.device)
                    X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)**(1 / self.problem.d)).to(self.device)
            elif self.problem.boundary == 'two_spheres':
                X = pt.randn(self.K_original, self.problem.d).to(self.device)
                X = self.problem.boundary_distance_2 * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K_original).unsqueeze(1)**(1 / self.problem.d)).to(self.device)
                selection = pt.sqrt(pt.sum(X**2, 1)) > self.problem.boundary_distance_1
                X = X[selection, :]
                self.K = int(pt.sum(selection))
                #X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                #X = X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K, self.problem.d).to(self.device) * (self.problem.boundary_distance_2 - self.problem.boundary_distance_1) + self.problem.boundary_distance_1)
            elif self.problem.boundary in ['square', 'unbounded_square']:
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l

            if 'unbounded' not in self.problem.boundary:
                t_n_boundary = pt.rand(self.K_boundary, 1).to(self.device) * self.problem.T
                X_t_n_boundary = pt.cat([X_boundary, t_n_boundary], 1)

            if self.loss_method not in ['BSDE-4', 'BSDE'] and self.boundary_loss:
                X_T = pt.cat([X[:self.K_boundary, :], self.problem.T * pt.ones(self.K_boundary).to(self.device).unsqueeze(1)], 1)
                loss += self.alpha[1] * pt.mean((self.V(X_T).squeeze() - self.problem.f(X[:self.K_boundary, :]))**2)
                if 'unbounded' not in self.problem.boundary:
                    if self.problem.boundary_type == 'Dirichlet':
                        loss += self.alpha[2] * pt.mean((self.V(X_t_n_boundary).squeeze() - self.problem.g(X_boundary, t_n_boundary.squeeze()))**2)
                    elif self.problem.boundary_type == 'Neumann':
                        X_t_n_boundary = pt.autograd.Variable(X_t_n_boundary, requires_grad=True)
                        Y_ = self.V(X_t_n_boundary)
                        Y_eval = Y_.squeeze().sum()
                        Y_eval.backward(retain_graph=True)
                        grad_V, = pt.autograd.grad(Y_eval, X_t_n_boundary, create_graph=True)
                        loss += self.alpha[2] * pt.mean((pt.sum(grad_V[:, :self.d] * X_boundary, 1) - pt.sum(self.problem.g(X_boundary, t_n_boundary.squeeze()) * X_boundary, 1))**2)

            X = pt.autograd.Variable(X, requires_grad=True)
            Y = pt.zeros(self.K).to(self.device)
            t_n = pt.rand(self.K, 1).to(self.device) * self.problem.T
            X_t_n = pt.cat([X, t_n], 1)
            if self.loss_method in ['BSDE-2', 'BSDE-4', 'BSDE', 'diffusion']:
                Y = self.V(X_t_n).squeeze()

            #lambda_log.append(lambda_(X)[0].item())
            stopped = pt.zeros(self.K).bool().to(self.device)
            hitting_times = pt.zeros(self.K)
            V_L2 = pt.zeros(self.K)
            K_count = 0

            #phi_0 = self.V(X_t_n).squeeze()

            for n in range(self.N):

                selection = ~stopped
                K_selection = pt.sum(selection)

                if K_selection == 0:
                    break

                if self.solve_linear_L2_projection is False:
                    Y_ = self.V(X_t_n)
                    Y_eval = Y_.squeeze().sum()
                    Y_eval.backward(retain_graph=True)
                    grad_V, = pt.autograd.grad(Y_eval, X, create_graph=True)
                    Z = pt.mm(self.problem.sigma(X).t(), grad_V.t()).t()

                xi = pt.randn(self.K, self.d).to(self.device)

                #V_L2[selection] += ((self.V(X_t_n[selection]).squeeze() - pt.tensor(self.problem.v_true(X[selection], t_n[selection])).float().squeeze())**2).detach().cpu() * self.delta_t_np

                c = pt.zeros(self.d, self.K).to(self.device)
                if self.adaptive_forward_process is True:
                    c = -Z.t()
                if self.detach_forward is True:
                    c = c.detach()

                X_proposal = (X + ((self.problem.b(X) + pt.mm(self.problem.sigma(X), c).t()) * self.delta_t
                     + pt.mm(self.problem.sigma(X), xi.t()).t() * self.sq_delta_t) * selection.float().unsqueeze(1).repeat(1, self.d))

                new_selection = pt.ones(self.K).bool().to(self.device)
                hitting_times[selection] += 1
                if self.problem.boundary == 'sphere':
                    new_selection = pt.all(pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) < self.problem.boundary_distance, 1).to(self.device)
                elif self.problem.boundary == 'two_spheres':
                    new_selection = ((pt.sqrt(pt.sum(X**2, 1)) > self.problem.boundary_distance_1) & (pt.sqrt(pt.sum(X**2, 1)) < self.problem.boundary_distance_2)).to(self.device)
                elif self.problem.boundary == 'square':
                    if self.problem.one_boundary:
                        new_selection = pt.any((X_proposal <= self.problem.X_r), 1).to(self.device)
                    else:
                        new_selection = pt.all((X_proposal >= self.problem.X_l) & (X_proposal <= self.problem.X_r), 1).to(self.device)

                new_selection = new_selection & ((t_n.squeeze() + self.delta_t) <= self.problem.T)

                if self.loss_method == 'BSDE-2':
                    loss += self.alpha[0] * pt.mean((Y_.squeeze() - Y)**2 * (new_selection & ~stopped).float())

                if self.solve_linear_L2_projection is False:
                    if self.loss_method in ['BSDE-2', 'BSDE-4']:
                        Y = (Y + ((- self.problem.h(n * self.delta_t, X, Y, Z) #- lambda_(X) * Y_.squeeze() #  lambda_(X) 
                                   + pt.sum(Z * c.t(), 1)) * self.delta_t + pt.sum(Z * xi, 1) * self.sq_delta_t) * (new_selection & ~stopped).float())
                    else:
                        Y = (Y + ((- self.problem.h(n * self.delta_t, X, Y_.squeeze(), Z) #- lambda_(X) * Y_.squeeze() #  lambda_(X) 
                                   + pt.sum(Z * c.t(), 1)) * self.delta_t + pt.sum(Z * xi, 1) * self.sq_delta_t) * (new_selection & ~stopped).float())

                X_ = X
                X = (X * (~new_selection | stopped).float().unsqueeze(1).repeat(1, self.d) 
                     + X_proposal * (new_selection & ~stopped).float().unsqueeze(1).repeat(1, self.d))

                t_n += self.delta_t * (new_selection & ~stopped).float().unsqueeze(1)
                X_t_n = pt.cat([X, t_n], 1)

                if self.loss_method in ['BSDE', 'diffusion']:
                    K_count += pt.sum(new_selection & ~stopped)

                if pt.sum(~new_selection & ~stopped) > 0:
                    stopped[~new_selection & ~stopped] += True

                if self.loss_method == 'BSDE-3':
                    loss += self.alpha[0] * pt.mean((self.V(X).squeeze() - Y_.squeeze() + (self.problem.h(X_, Y_.squeeze(), Z)
                                                                 - pt.sum(Z * c.t(), 1)) * self.delta_t 
                                     - pt.sum(Z * xi, 1) * self.sq_delta_t)**2 * (new_selection & ~stopped).float())

            if self.loss_method == 'diffusion':
                loss += self.alpha[0] * pt.mean((self.V(X_t_n).squeeze() - Y)**2)
            self.V.zero_grad()
            #lambda_.zero_grad()

            if self.loss_method in ['BSDE', 'diffusion']:
                self.K_log.append(K_count.item())

            if self.loss_method in ['BSDE-4', 'BSDE']:
                if pt.sum(stopped).item() != self.K:
                    print('Not all trajectories stopped.')
                if 'unbounded' in self.problem.boundary:
                    loss += pt.mean((Y - self.problem.f(X))**2)
                elif self.problem.boundary_type == 'Dirichlet':
                    loss += pt.mean((Y - self.problem.g(X, t_n.squeeze()))**2)
                elif self.problem.boundary_type == 'Neumann':
                    T_selection = (t_n > (self.problem.T - self.delta_t)).squeeze()
                    if pt.sum(T_selection) > 0:
                        #loss += pt.mean((self.V(X_t_n[T_selection, :]).squeeze() - self.problem.f(X[T_selection, :]))**2)
                        loss += pt.mean((Y[T_selection] - self.problem.f(X[T_selection, :]))**2)
                    if pt.sum(T_selection) < self.K:
                        loss += pt.mean((pt.sum(grad_V * X, 1) - pt.sum(self.problem.g(X, t_n.squeeze()) * X, 1))**2)

            if self.loss_with_stopped:
                loss += pt.mean((Y[stopped] - self.problem.f(X[stopped, :]))**2) # self.alpha[0] *
            loss.backward(retain_graph=True)
            self.V.optim.step()

            #lambda_.optim.step()

            self.loss_log.append(loss.item())
            self.V_L2_log.append(pt.mean(V_L2).item())
            if self.K_test_log is not None:
                L2_error, mean_abolute_error, mean_relative_error = compute_test_error(self, self.problem, self.K_test_log, self.device, 'parabolic')
                self.V_test_L2.append(L2_error)
                self.V_test_abs.append(mean_abolute_error)
                self.V_test_rel_abs.append(mean_relative_error)

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if self.verbose:
                if l % self.print_every == 0:
                    print('%d - loss = %.4e, v L2 error = %.4e, n = %d, active: %d/%d, %.2f' % 
                          (l, self.loss_log[-1], self.V_L2_log[-1], n, K_selection, self.K, np.mean(self.times[-self.print_every:])))

    def train_PINN(self):
        for l in range(self.L):

            t_0 = time.time()

            if self.problem.boundary == 'sphere':
                X_boundary = pt.randn(self.K_boundary, self.d).to(self.device)
                X_boundary = self.problem.boundary_distance * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1)
            elif self.problem.boundary == 'two_spheres':
                X_boundary = pt.randn(self.K_boundary, self.problem.d).to(self.device)
                X_boundary = (pt.tensor([self.problem.boundary_distance_1] * int(self.K_boundary / 2) + 
                                        [self.problem.boundary_distance_2] * int(self.K_boundary / 2)).unsqueeze(1).to(self.device)
                              * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1))
            elif self.problem.boundary == 'square':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_l
                X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_l
                X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_r
                if self.problem.one_boundary:
                    X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_r
                    X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_r
            elif self.problem.boundary == 'square-corner':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_corner) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_corner
                X_boundary[pt.tensor(a.astype(float)).bool()] = self.problem.X_corner
                X_boundary[pt.tensor(b.astype(float)).bool()] = self.problem.X_corner

            if self.problem.boundary in ['sphere', 'unbounded']:
                if self.uniform_square:
                    X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                    X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)).to(self.device)
                else:
                    X = pt.randn(self.K, self.problem.d).to(self.device)
                    X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)**(1 / self.problem.d)).to(self.device)
            elif self.problem.boundary == 'two_spheres':
                X = pt.randn(self.K_original, self.problem.d).to(self.device)
                X = self.problem.boundary_distance_2 * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K_original).unsqueeze(1)**(1 / self.problem.d)).to(self.device)
                selection = pt.sqrt(pt.sum(X**2, 1)) > self.problem.boundary_distance_1
                X = X[selection, :]
                self.K = int(pt.sum(selection))
                #X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                #X = X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K, self.problem.d).to(self.device) * (self.problem.boundary_distance_2 - self.problem.boundary_distance_1) + self.problem.boundary_distance_1)
            elif self.problem.boundary in ['square', 'unbounded_square']:
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l
            elif self.problem.boundary == 'square-corner':
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l
                X[pt.all(X > self.problem.X_corner, 1), :] = -X[pt.all(X > self.problem.X_corner, 1), :]

            t_n = pt.rand(self.K, 1).to(self.device) * self.problem.T
            X_t_n = pt.cat([X, t_n], 1)

            second_derivatives = pt.zeros(X.shape[0]).to(self.device)

            X = pt.autograd.Variable(X, requires_grad=True)
            X_t_n = pt.autograd.Variable(X_t_n, requires_grad=True)

            V_eval = self.V(X_t_n).squeeze()
            grad = pt.autograd.grad(V_eval, X_t_n, grad_outputs=pt.ones(self.K).to(self.device), create_graph=True)[0]

            if self.full_hessian:
                for i, x in enumerate(X):
                    hess = pt.autograd.functional.hessian(self.V, x.unsqueeze(0), create_graph=True).squeeze()
                    second_derivatives[i] = pt.sum(pt.diagonal(pt.mm(pt.mm(self.problem.B, self.problem.B.t()), hess)))
            else:
                for k in range(grad.shape[1]):
                    rad_grad_u_xx = pt.autograd.grad(grad[:, k], X_t_n, grad_outputs=pt.ones(self.K).to(self.device), create_graph=True)[0][:, k]
                    if k != self.d:
                        second_derivatives += rad_grad_u_xx
                second_derivatives = self.problem.B[0, 0]**2 * second_derivatives

            loss = self.alpha[0] * pt.mean((grad[:, self.d:].squeeze() + 0.5 * second_derivatives + pt.sum(self.problem.b(X) * grad[:, :self.d], 1) 
                                                + self.problem.h(t_n, X, self.V(X_t_n).squeeze(), pt.mm(self.problem.B, grad[:, :self.d].t()).t()))**2) 
            if self.log_loss_parts:
                self.loss_log_domain.append(loss.item() / self.alpha[0])
            if self.boundary_loss:
                if 'unbounded' not in self.problem.boundary:
                    t_n_boundary = pt.rand(self.K_boundary, 1).to(self.device) * self.problem.T
                    X_t_n_boundary = pt.cat([X_boundary, t_n_boundary], 1)
                X_T = pt.cat([X[:self.K_boundary, :], self.problem.T * pt.ones(self.K_boundary).to(self.device).unsqueeze(1)], 1)
                loss += self.alpha[1] * pt.mean((self.V(X_T).squeeze() - self.problem.f(X[:self.K_boundary, :]))**2)
                if 'unbounded' not in self.problem.boundary:
                    if self.problem.boundary_type == 'Dirichlet':
                        loss += self.alpha[2] * pt.mean((self.V(X_t_n_boundary).squeeze() - self.problem.g(X_boundary, t_n_boundary.squeeze()))**2)
                    elif self.problem.boundary_type == 'Neumann':
                        X_t_n_boundary = pt.autograd.Variable(X_t_n_boundary, requires_grad=True)
                        Y_ = self.V(X_t_n_boundary)
                        Y_eval = Y_.squeeze().sum()
                        Y_eval.backward(retain_graph=True)
                        grad_V, = pt.autograd.grad(Y_eval, X_t_n_boundary, create_graph=True)
                        loss += self.alpha[2] * pt.mean((pt.sum(grad_V[:, :self.d] * X_boundary, 1) - pt.sum(self.problem.g(X_boundary, t_n_boundary.squeeze()) * X_boundary, 1))**2)

            self.V.zero_grad()

            loss.backward(retain_graph=True)
            self.V.optim.step()

            #self.V_L2_log.append(pt.mean(((self.V(X_t_n).squeeze() - pt.tensor(self.problem.v_true(X_t_n.detach())).float().squeeze())**2).detach().cpu() * self.delta_t_np).item())
            self.V_L2_log.append(0)
            self.loss_log.append(loss.item())
            if self.K_test_log is not None:
                L2_error, mean_abolute_error, mean_relative_error = compute_test_error(self, self.problem, self.K_test_log, self.device, 'parabolic')
                self.V_test_L2.append(L2_error)
                self.V_test_abs.append(mean_abolute_error)
                self.V_test_rel_abs.append(mean_relative_error)

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                print('%d - loss = %.4e - v L2 error = %.4e - %.2f' % (l, self.loss_log[-1], self.V_L2_log[-1], np.mean(self.times[-self.print_every:])))
