import torch
import math
import random

from .basealgorithm import BaseOptimizer

class FedpaqOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        self.lr = kwargs.get('lr')
        self.momentum = kwargs.get('momentum', 0.)
        defaults = dict(lr=self.lr, momentum=self.momentum)
        super(FedpaqOptimizer, self).__init__(params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta = param.grad.data
                if beta > 0.:
                    if 'momentum_buffer' not in self.state[param]:
                        self.state[param]['momentum_buffer'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer'].mul_(beta).add_(delta.mul(1. - beta)) # \beta * v + (1 - \beta) * grad
                    delta = self.state[param]['momentum_buffer']
                param.data.sub_(delta)
        return loss
    
    def quantize(self, x,input_compress_settings={}):
        compress_settings={'n':6}
        compress_settings.update(input_compress_settings)
        #assume that x is a torch tensor
        
        n=compress_settings['n']
        #print('n:{}'.format(n))
        x=x.float()
        x_norm=torch.norm(x,p=float('inf'))
        
        sgn_x=((x>0).float()-0.5)*2
        
        p=torch.div(torch.abs(x),x_norm)
        renormalize_p=torch.mul(p,n)
        floor_p=torch.floor(renormalize_p)
        compare=torch.rand_like(floor_p)
        final_p=renormalize_p-floor_p
        margin=(compare < final_p).float()
        xi=(floor_p+margin)/n
        
        Tilde_x=x_norm*sgn_x*xi

        return Tilde_x
    
    def simulated_annealing(self, x, B):
        d = len(x)

        # Sort indices by x_j descending
        indices = torch.argsort(-x)

        # Initial solution: allocate b based on sorted indices
        b = torch.zeros(d, dtype=torch.int)
        for i in range(B // 2):
            i = i % d
            b[indices[i]] += 2

        def objective(b):
            return torch.sum(x**2 / 4**b)

        # Parameters
        T = 1000.0
        alpha = 0.95
        min_T = 1e-8
        max_iter = 100

        current_value = objective(b)
        best_b = b.clone()
        best_value = current_value

        while T > min_T and max_iter > 0:
            # Generate new solution
            new_b = b.clone()

            # Choose multiple indices to adjust
            num_changes = random.randint(1, d // 2)
            selected_indices = random.sample(range(d), num_changes)

            for i in selected_indices:
                if new_b[indices[i]] > 0:
                    new_b[indices[i]] -= 2
                    # Choose another index to increase
                    j = random.choice(range(d))
                    if new_b[indices[j]] < 8:
                        new_b[indices[j]] += 2

            # Ensure b only contains 0, 2, 4, 8
            new_b = torch.clamp(new_b, 0, 8)

            # Check if new_b is valid
            if new_b.sum() == B:
                new_value = objective(new_b)
                delta = new_value - current_value

                # Accept new solution with probability
                if delta < 0 or random.uniform(0, 1) < math.exp(-delta / T):
                    b = new_b
                    current_value = new_value

                    if current_value < best_value:
                        best_b = b.clone()
                        best_value = current_value

            # Cool down
            T *= alpha
            max_iter -= 1

        return best_b

    def accumulate(self, mixing_coefficient, local_param_iterator, partial_agg_condition=lambda name: None):
        for group in self.param_groups:
            for server_param, (name, local_param) in zip(group['params'], local_param_iterator):
                if partial_agg_condition(name):
                    continue
                vec = server_param.data - local_param.data
                ori_shape = vec.shape
                tensor = vec.view(-1)
                vec_q= torch.zeros_like(tensor)
                vec_q=self.quantize(tensor, input_compress_settings={'n':4})
                vec = vec_q.reshape(ori_shape)                
                local_delta = vec.mul(mixing_coefficient)
                if server_param.grad is None: # NOTE: grad buffer is used to accumulate local updates!
                    server_param.grad = local_delta
                else:
                    server_param.grad.add_(local_delta)
