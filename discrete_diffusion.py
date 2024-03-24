import torch
import torch.nn.functional as F
import math

EPS = 1e-30
LOG_EPS = math.log(EPS)

# ----------------------------------------- probability helper ---------------------------------------- # 
def sample_uniform_categorical(x_shape, num_classes, device="cuda"):
    return torch.randint(num_classes, size=x_shape, device = device)

def sample_bernoulli(prob, x_shape, device="cuda"):
    assert prob.shape[0] == x_shape[0]
    u = torch.rand(x_shape, device=device)
    b = u.clamp(min=EPS) < prob
    return b

def sample_categorical(prob):
    ind_sample = torch.multinomial(prob.flatten(end_dim=-2), num_samples=1).view(prob.shape[:-1])
    return ind_sample # has the same device as prob 

def logits_to_prob(logits, dim=-1):
    return F.softmax(logits, dim=dim)

def logits_to_logprob(logits, dim=-1):
    return F.log_softmax(logits, dim=dim)

# ----------------------------------------- index helper ---------------------------------------------- # 
def get_broadcast_idx(shape):
    return [torch.arange(s).view([1]*i+[-1]+[1]*(len(shape)-1-i)) for i, s in enumerate(shape)]

def index_last_dim(x, idx):
    """ Index the last dimension of the input tensor `x` using the indices `idx`.
    Args:
        x  : Tensor of shape (B, N1, ..., Nk, C).
        idx: Tensor of shape (B, N1, ..., Nk) with values less than C.
    Returns:
        Tensor of shape (B, N1, ..., Nk) containing the indexed values.
    """
    assert idx.max() < x.shape[-1]
    broadcast_idx = get_broadcast_idx(idx.shape)
    return x[broadcast_idx + [idx]]

def set_last_dim(x, idx, value=0, inplace_add=False):
    """ Set the last dimension of the input tensor `x` at the specified indices `idx` to the given `value`.
    Args:
        x   : The input tensor of shape (B, N1, ..., Nk, C).
        idx : The indices tensor of shape (B, N1, ..., Nk) with idx.max() < C.
        value (torch.Tensor or scalar, optional): The value to set at the specified indices. Defaults to 0.
        inplace_add (bool, optional): If True, adds the value to the existing tensor at the specified indices. 
                                      If False, replaces the tensor values at the specified indices with the value. 
                                      Defaults to False.
    Returns:
        torch.Tensor: The modified tensor of shape (B, N1, ..., Nk, C).
    """
    assert idx.max() < x.shape[-1]
    broadcast_idx = get_broadcast_idx(idx.shape)
    if inplace_add:
        x[broadcast_idx + [idx]] += value
    else:
        x[broadcast_idx + [idx]] = value 
    return x

# ----------------------------------------- diffusion components ---------------------------------------- # 
def noise_schedule(t_step, 
                   s_step=None,
                   schedule_type:str= "cosine",
                   N:int = 1000, # N=0 means continuous 
                   Tmax:float =1,
                   a:float=None, b:float=None, 
                   min_alphabar:float=1e-10, max_beta:float=100, 
                   **kwargs):

    assert t_step.max() <= Tmax if N == 0 else t_step.max() <= N
    step_to_time = lambda step: step if N == 0 else step/N * Tmax
    t = step_to_time(t_step)
    s = torch.tensor(0.0) if s_step is None else step_to_time(s_step)

    if schedule_type == "cosine":      
        a = a or 0.008            # set default value
        h = lambda t: torch.cos((t/Tmax + a)/ (1+a) * torch.pi * 0.5)
        h_t, h_s = h(t), h(s) 
        alphabar_t = h_t / h_s
        beta_t = torch.pi * torch.tan((t/Tmax + a) / (1 + a) * torch.pi * 0.5)
        beta_t = beta_t / (2*Tmax*(1+a))
    elif schedule_type == "exponential":
        a, b = a or 0.5, b or 10  # set default value
        b_power = lambda t: torch.exp(t/Tmax * math.log(b))
        b_power_t, b_power_s = b_power(t), b_power(s)
        alphabar_t = torch.exp(a * t * (b_power_s - b_power_t))
        beta_t = a * b_power_t * math.log(b)  
    elif schedule_type == "linear":
        alphabar_t = 1 - t/Tmax
        beta_t = 1/(Tmax - t)
    elif schedule_type == "constant":
        a = a or 0.03
        h = lambda t: torch.exp(-a * t)
        h_t, h_s = h(t), h(s) 
        alphabar_t = h_t / h_s
        beta_t = torch.full_like(t, a) 
    else:
        raise NotImplementedError

    assert alphabar_t.dim() == 1 
    alphabar_t = torch.clip(alphabar_t, min=min_alphabar, max=1-min_alphabar)
    beta_t = torch.clip(beta_t, max=max_beta)  # TODO: revise later

    return alphabar_t, beta_t

class UnifiedDiscreteDiffusion:
    """
    Unified discrete-time and continuous-time discrete diffusion model.
    The code is closely following the paper's notation and implementation.
    Understanding the paper's equations is necessary to follow the code.
    We work with categorical labels directly, instead of one-hot representation. 

    - Notice that the 'conditional_mask' argument is used for both conditioning and padding part. 
    """
    def __init__(self, num_steps, num_classes, noise_schedule_type, noise_schedule_args):
        self.num_steps = num_steps                      # 0 indicates using continuous-time diffusion 
        self.num_classes = num_classes
        self.noise_schedule_type = noise_schedule_type
        self.noise_schedule_args = noise_schedule_args  # args passed to noise_schedule

    @torch.no_grad()
    def get_alphabar_beta(self, t, s=None): 
        """
        t: (B,)
        """
        alphabar_t, beta_t = noise_schedule(t, s, 
                                            schedule_type=self.noise_schedule_type, 
                                            N=self.num_steps, **self.noise_schedule_args)
        return alphabar_t, beta_t

    @torch.no_grad()
    def get_m_dot_xt(self, x_t, m=None): 
        """
        x_t: (B, N1, ..., Nk)
        m  : (B, N1, ..., Nk, C) or None or (C)
        """
        if m is None:
            m_dot_xt = torch.full_like(x_t, 1/self.num_classes, dtype=torch.float32) 
        elif m.dim() == 1:
            m_dot_xt = m[x_t]
        else:
            m_dot_xt = index_last_dim(m, x_t) 
        return m_dot_xt
    
    @torch.no_grad()
    def get_lambda(self, alphabar_t, alphabar_s, x_t, m=None):  
        """
        alphabar_t: (B,1, ..., 1k)
        alphabar_s: (B,1, ..., 1k)
        x_t       : (B, N1, ..., Nk)
        m         : (B, N1, ..., Nk, C) or None or (C)
        return    : (B, N1, ..., Nk)
        """
        assert x_t.dim() == alphabar_t.dim() 
        alpharbar_t_s = alphabar_t / alphabar_s
        m_dot_xt = self.get_m_dot_xt(x_t, m) # (B, N1, ..., Nk)            
        lambda_t_s = (1-alphabar_s)*(1-alpharbar_t_s) * m_dot_xt
        lambda_t_s = lambda_t_s / (alphabar_t + (1-alphabar_t)*m_dot_xt)
        assert (lambda_t_s.shape == x_t.shape)
        return lambda_t_s 
    
    @torch.no_grad()
    def get_mu(self, alphabar_t, alphabar_s): 
        """
        alphabar_t: (B,1,...,1k)
        return    : (B,1,...,1k)
        """
        mu_t_s = (1-alphabar_s) / (1-alphabar_t)
        return mu_t_s

    @torch.no_grad()
    def get_mu_times_alphabar(self, alphabar_t, alphabar_s): 
        mul = alphabar_s * alphabar_t
        return  (alphabar_t - mul)/(alphabar_s - mul)

    @torch.no_grad()
    def get_gamma_coef(self, alphabar_t, alphabar_s, x_t, m=None): 
        # mu - lambda - mu*alphabar
        m_dot_xt = self.get_m_dot_xt(x_t, m) # (B, N1, ..., Nk)  
        coef = self.get_mu_times_alphabar(alphabar_t, alphabar_s) * (alphabar_s - alphabar_t)
        coef = coef / (alphabar_t + (1-alphabar_t)*m_dot_xt) # coef > 0
        assert coef.shape == x_t.shape
        return coef
    
    @torch.no_grad()
    def qt_0_sample(self, x_0, t, m=None, conditional_mask=None):
        """ forward sampling
        x_0: (B, N1, ..., Nk)
        t  : (B,)
        m  : (B, N1, ..., Nk, C), or None, or (C)
        conditional_mask : (B, N1, ..., Nk) or None
        """
        assert x_0.dim() >= 2
        sample_shape = x_0.shape
        alphabar_t, _ = self.get_alphabar_beta(t)  
        alphabar_t = alphabar_t.view([-1]+[1]*(x_0.dim()-1)) #B,N1,....Nk

        #fast sampling from Cat(m)
        if m is None:
            m0 = sample_uniform_categorical(sample_shape, self.num_classes, device=x_0.device) # x_0 shape
        elif m.dim() == 1:
            # sample B x N1 x ... x Nk times with replacement. 
            m0 = torch.multinomial(m, num_samples=sample_shape.numel(), replacement=True).view(sample_shape)
        else:
            assert m.shape[:-1] == sample_shape and m.shape[-1] == self.num_classes
            m0 = sample_categorical(m)
        #sample from the branch indicator function
        bt = sample_bernoulli(alphabar_t, sample_shape, device=x_0.device) # x_0 shape
        #sample of size BxD
        sample = torch.where(bt, x_0, m0)
        if conditional_mask is not None:
            assert conditional_mask.shape == x_0.shape
            sample[conditional_mask] = x_0[conditional_mask]
        return sample
    
    @torch.no_grad() ### ONLY used in continuous-time case
    def qt_0_prob(self, x_0, t, m=None, return_beta=False):
        """
        x_0: (B, N1, ..., Nk)
        t  : (B,)
        m  : (B, N1, ..., Nk, C) or None or (C)
        """
        shape = [-1]+[1]*(x_0.dim()) #B, 1, ..., 1_k, 1
        alphabar_t, beta_t = self.get_alphabar_beta(t)
        alphabar_t, beta_t = alphabar_t.view(shape), beta_t.view(shape)
        if m is None:
            m = torch.full_like(x_0, 1/self.num_classes, dtype=torch.float32).unsqueeze(-1).repeat_interleave(self.num_classes,-1)
        elif m.dim() == 1:
            m = torch.broadcast_to(m, list(x_0.shape)+[self.num_classes])
        
        prob = (1-alphabar_t) * m
        prob = set_last_dim(prob, x_0, value=alphabar_t.squeeze(-1), inplace_add=True)
        if return_beta:
            return prob, beta_t
        return prob

    @torch.no_grad()
    def qs_t0_prob(self, x_t, x_0, t, s, m=None):
        """ forward prob, requires s > 0, t > s
        x_0: (B, N1, ..., Nk)
        x_t: (B, N1, ..., Nk)
        s  : (B,) > 0 
        t  : (B,) > s
        m  : (B, N1, ..., Nk, C) or None, or (C)
        """
        shape = [-1]+[1]*(x_0.dim()-1) #B,N1,....Nk
        alphabar_t, _ = self.get_alphabar_beta(t)
        alphabar_s, _ = self.get_alphabar_beta(s)
        alphabar_t, alphabar_s = alphabar_t.view(shape), alphabar_s.view(shape)

        mu_alphabar_t_s = self.get_mu_times_alphabar(alphabar_t, alphabar_s)
        mu_t_s = self.get_mu(alphabar_t, alphabar_s)
        lambda_t_s = self.get_lambda(alphabar_t, alphabar_s, x_t, m) # (B, N1, ..., Nk)

        # compute x_0=x_t prob 
        prob_eq = lambda_t_s[..., None] 
        prob_eq = prob_eq * m if m is not None else (prob_eq/self.num_classes).repeat_interleave(self.num_classes,-1) # (B, N1, ..., Nk, C)
        broadcast_idx = get_broadcast_idx(x_t.shape)
        prob_eq[broadcast_idx+[x_t]] += 1 - lambda_t_s

        # compute x_0!=x_t prob
        prob_neq = (mu_t_s - mu_alphabar_t_s)[...,None]                             # (B, 1, ..., 1k, 1) 
        prob_neq = prob_neq * m if m is not None else prob_neq / self.num_classes   # (B, 1, ..., 1k, C) or (B,N1...Nk,C)
        prob_neq = torch.broadcast_to(prob_neq, list(x_t.shape)+[self.num_classes]).clone() # (B, N1, ..., Nk, C)

        prob_neq[broadcast_idx+[x_t]] += torch.broadcast_to(mu_alphabar_t_s, x_t.shape) # mu'shape is (B,1,...,1k)
        prob_neq[broadcast_idx+[x_0]] += torch.broadcast_to(1-mu_t_s       , x_0.shape)
        
        prob = torch.where((x_t==x_0).unsqueeze(-1), prob_eq, prob_neq)
        return prob 

    def ps_t_prob(self, fprob_t, x_t, t, s, m=None):
        """ Backward prob, requires s > 0, t > s
        fprob_t: (B, N1, ..., Nk, C)
        x_t    : (B, N1, ..., Nk)
        s      : (B,)
        t      : (B,)
        m      : (B, N1, ..., Nk, C) or None or (C)
        """
        shape = [-1]+[1]*(x_t.dim()-1) #B,N1,....Nk
        alphabar_t, _ = self.get_alphabar_beta(t)
        alphabar_s, _ = self.get_alphabar_beta(s)
        alphabar_t, alphabar_s = alphabar_t.view(shape), alphabar_s.view(shape)

        mu_alphabar_t_s = self.get_mu_times_alphabar(alphabar_t, alphabar_s)
        mu_t_s = self.get_mu(alphabar_t, alphabar_s)
        gamma_t_s = self.get_gamma_coef(alphabar_t, alphabar_s, x_t, m) * index_last_dim(fprob_t, x_t)

        prob = (mu_t_s - mu_alphabar_t_s - gamma_t_s)[..., None]
        prob = prob * m if m is not None else (prob/self.num_classes)#.repeat_interleave(self.num_classes,-1)
        prob = torch.broadcast_to(prob, list(x_t.shape)+[self.num_classes])

        prob = prob + (1 - mu_t_s[..., None]) * fprob_t 
        broadcast_idx = get_broadcast_idx(x_t.shape)
        prob[broadcast_idx+[x_t]] = prob[broadcast_idx+[x_t]] + gamma_t_s + mu_alphabar_t_s

        prob = torch.clip(prob, min=0)          # make sure prob >= 0 
        return prob 
    
    def ps_t_logprob(self, flogprob_t, x_t, t, s, m=None):
        """ Backward logprob, requires s > 0, t > s
        fprob_t: (B, N1, ..., Nk, C)
        x_t    : (B, N1, ..., Nk)
        s      : (B,)
        t      : (B,)
        m      : (B, N1, ..., Nk, C) or None or (C)
        """
        shape = [-1]+[1]*(x_t.dim()-1) #B,N1,....Nk
        alphabar_t, _ = self.get_alphabar_beta(t)
        alphabar_s, _ = self.get_alphabar_beta(s)
        alphabar_t, alphabar_s = alphabar_t.view(shape), alphabar_s.view(shape)

        mu_alphabar_t_s = self.get_mu_times_alphabar(alphabar_t, alphabar_s)
        mu_t_s = self.get_mu(alphabar_t, alphabar_s)

        # term 1 
        logterm1 = flogprob_t + (1-mu_t_s).clip(min=EPS).log()[...,None]
        # compute fprob_dot_xt
        gamma_t_s = self.get_gamma_coef(alphabar_t, alphabar_s, x_t, m) * (index_last_dim(flogprob_t, x_t).exp())
        probterm2 = (mu_t_s - mu_alphabar_t_s - gamma_t_s)[..., None]
        probterm2 = probterm2 * m if m is not None else probterm2/self.num_classes #).repeat_interleave(self.num_classes,-1)
        probterm2 = torch.broadcast_to(probterm2, list(x_t.shape)+[self.num_classes]).clone()
        broadcast_idx = get_broadcast_idx(x_t.shape)
        probterm2[broadcast_idx+[x_t]] = probterm2[broadcast_idx+[x_t]] + gamma_t_s + mu_alphabar_t_s

        logprob = torch.logaddexp(logterm1, probterm2.clip(min=EPS).log())
        return logprob
    
    def ps_t0_delta(self, fprob_t, x_t, x_0, t, s, m=None):
        """ ps_t  - qs_t_0, requires s > 0, t > s
        fprob_t   : (B, N1, ..., Nk, C)
        x_t       : (B, N1, ..., Nk)
        x_0       : (B, N1, ..., Nk)
        t         : (B,)
        s         : (B,)
        m         : (B, N1, ..., Nk, C) or None or (C) 
        """
        shape = [-1]+[1]*(x_t.dim()-1) #B,1_1,....1_k
        alphabar_t, _ = self.get_alphabar_beta(t)
        alphabar_s, _ = self.get_alphabar_beta(s)
        alphabar_t, alphabar_s = alphabar_t.view(shape), alphabar_s.view(shape)

        # mu_t_s = self.get_mu(alphabar_t, alphabar_s)
        f_minus_x0 = fprob_t - F.one_hot(x_0, num_classes=self.num_classes)
        xt_dot_m = self.get_m_dot_xt(x_t, m) # (B, N1, ..., Nk)
        m = torch.broadcast_to(m, fprob_t.shape) if m is not None else torch.full_like(fprob_t, 1/self.num_classes)
        xt_minus_m = F.one_hot(x_t, num_classes=self.num_classes) - m

        phi_t_s = (1-alphabar_s) * alphabar_t / alphabar_s 
        phi_t_s = phi_t_s / (alphabar_t + (1-alphabar_t)*xt_dot_m)
        # clip the value to avoid numerical issue
        phi_t_s = torch.clip(phi_t_s, max=1)

        delta_without_coef = f_minus_x0 + (phi_t_s*index_last_dim(f_minus_x0, x_t))[...,None] * xt_minus_m
        return delta_without_coef
    
    # ----------------------------------------- loss computation: discrete-time case  ---------------------------------------- #
    def _prior_at_T(self, x_0, m=None):
        """ Compute KL(q(x_T | x_0) || p(x_T)). This part is not useful in computing gradient, but useful for evaluation.
        x_0: (B, N1, ..., Nk)
        t  : (B,)
        m  : (B, N1, ..., Nk, C) or None, or (C)
        """
        batch_size = x_0.size(0)
        T = self.num_steps * torch.ones(batch_size, device=x_0.device) # (B,)
        qT_0_prob = self.qt_0_prob(x_0, T, m=m) # (B, N1, ..., Nk, C)
        pT_prob = torch.broadcast_to(m, qT_0_prob.shape) if m is not None else torch.full_like(qT_0_prob, 1/self.num_classes)
        return F.kl_div(pT_prob.clip(min=EPS).log(), qT_0_prob, reduction='none').sum(-1) # (B, N1, ..., Nk)
    
    def discrete_time_loss(self, flogits_t, x_t, x_0, t, m=None, conditional_mask=None, simplified_vlb=False):
        """
        conditional_mask : (B, N1, ..., Nk) or None, the mask is used for conditioning or padding. 
        flogits_t: (B, N1, ..., Nk, C)
        x_t      : (B, N1, ..., Nk)
        x_0      : (B, N1, ..., Nk)
        t        : (B,)
        m        : (B, N1, ..., Nk, C) or None, or (C)
        """
        batch_size = x_0.size(0)
        flogprob_t = logits_to_logprob(flogits_t)

        # CE loss
        ce_loss = -index_last_dim(flogprob_t, x_0)

        if simplified_vlb:
            # Approximated VLB with l2 loss for t>= 2
            delta_p_theta = self.delta_p_theta(logits_to_prob(flogits_t), x_t, x_0, t, s=t-1, m=m)        
            vlb_loss = (delta_p_theta**2).sum(-1)   
        else:
            # Exact vlb loss for t >= 2
            assert t.min() >= 1
            ps_t_logprob = self.ps_t_logprob(flogprob_t, x_t, t=t, s=t-1, m=m)
            qs_t0_prob = self.qs_t0_prob(x_t, x_0, t=t, s=t-1, m=m)
            vlb_loss = F.kl_div(ps_t_logprob, qs_t0_prob, reduction='none').sum(-1) 
        
        t0_mask = (t == 1).float().view([-1]+[1]*(x_0.dim()-1)) # min time is 1 
        vlb_loss = t0_mask * ce_loss + (1.0-t0_mask) * vlb_loss # this is not the final vlb, we still need prior loss for discrete case. 
        # prior loss 
        vlb_loss = vlb_loss + (self._prior_at_T(x_0, m=m) / self.num_steps)
        
        if conditional_mask is not None:
            assert conditional_mask.shape == x_0.shape
            assert (x_t==x_0)[conditional_mask].all()
            vlb_loss = vlb_loss * (~conditional_mask)
            ce_loss = ce_loss * (~conditional_mask)
            assert vlb_loss.shape == x_0.shape

        vlb_loss = vlb_loss.view(batch_size, -1).sum(-1)
        ce_loss = ce_loss.view(batch_size, -1).sum(-1)
        return vlb_loss.sum(), ce_loss.sum() # add all loss togther 
    
    # ----------------------------------------- loss computation: continuous-time case  ---------------------------------------- #
    def _log_gt_inner(self, flogprob_t, x_t, t, m=None, coef=1):
        """ Proposition 4's inner part's log. This is a much stable version of log. 
        fprob_t  : (B, N1, ..., Nk, C)
        x_t      : (B, N1, ..., Nk)
        t        : (B,)
        m        : (B, N1, ..., Nk, C) or None or (C)
        coef     : scalar, default is 1
        """
        m_dot_xt = self.get_m_dot_xt(x_t, m) # (B, N1, ..., Nk)
        alphabar_t, beta_t = self.get_alphabar_beta(t)
        alphabar_t = alphabar_t.view([-1]+[1]*(x_t.dim()-1)) #B,N1,....Nk
        fprob_t_dot_xt = index_last_dim(flogprob_t, x_t).exp()  #B,N1,....Nk
        term1 = (coef - (alphabar_t * fprob_t_dot_xt) / (alphabar_t + (1-alphabar_t) * m_dot_xt))[...,None]
        term1 = term1 * m if m is not None else term1/self.num_classes
        logterm1 = term1.clip(min=EPS).log()
        logterm2 = (alphabar_t.clip(min=EPS).log() - (1 - alphabar_t).clip(min=EPS).log())[...,None] + flogprob_t
        log_gt_inner = torch.logaddexp(logterm1, logterm2)
        set_last_dim(log_gt_inner, x_t, 0) # should be LOG_EXP, however the value should be set as 0 for loss
        return log_gt_inner
    
    def _gt_inner(self, fprob_t, x_t, t, m=None, coef=1):
        """ Proposition 4's inner part, this ignores the coefficient of gt. 
        fprob_t  : (B, N1, ..., Nk, C)
        x_t      : (B, N1, ..., Nk)
        t        : (B,)
        m        : (B, N1, ..., Nk, C) or None or (C)
        coef     : scalar, default is 1
        """
        alphabar_t, beta_t = self.get_alphabar_beta(t)
        alphabar_t = alphabar_t.view([-1]+[1]*(x_t.dim()-1)) #B,N1,....Nk
                        
        fprob_t_dot_xt = index_last_dim(fprob_t, x_t) #B,N1,....Nk
        m_dot_xt = self.get_m_dot_xt(x_t, m) # (B, N1, ..., Nk)
        inside_gt1 = (coef - (alphabar_t * fprob_t_dot_xt) / (alphabar_t + (1-alphabar_t) * m_dot_xt))[...,None]
        inside_gt1 = inside_gt1 * m if m is not None else inside_gt1/self.num_classes

        inside_gt2 = (alphabar_t / (1-alphabar_t))[..., None] * fprob_t
        inside_gt =  inside_gt2 + inside_gt1
        inside_gt = set_last_dim(inside_gt, x_t, value=1) ### change this to 1 for certain case of taking log

        # when coef =2, it is useful for MCMC corrector step (before multiply beta and step size)
        return inside_gt, beta_t, m_dot_xt
    
    def continuous_time_loss(self, flogits_t, x_t, x_0, t, m=None, conditional_mask=None, denoising_fn=None, uniform_sampling=False, simplified_vlb=False):
        """
        conditional_mask : (B, N1, ..., Nk) or None, the mask is used for conditioning or padding. 
        flogits_t: (B, N1, ..., Nk, C)
        x_t      : (B, N1, ..., Nk)
        x_0      : (B, N1, ..., Nk)
        t        : (B,)
        m        : (B, N1, ..., Nk, C) or None, or (C)
        """
        shape = [-1]+ [1]*(x_t.dim()-1) #B,1,....1_k
        m = torch.broadcast_to(m, fprob_t.shape) if m is not None else torch.full_like(fprob_t, 1/self.num_classes)
        ## ----------------- get  first term --------------------
        fprob_t = logits_to_prob(flogits_t) # (B, N1, ..., Nk, C)
        inside_gt, beta_t, m_dot_xt = self._gt_inner(fprob_t, x_t, t, m=m, coef=1.0)

        beta_t = beta_t.view(shape)
        beta_t_ori = beta_t.clone()
        if simplified_vlb:
            beta_t = torch.clip(beta_t, max=1.0)
        vlb_term1 = beta_t * inside_gt.sum(-1) # (B, N1, ..., Nk)
        
        ## ----------------- get second term --------------------
        B, D = x_t.size(0), x_t.size(-1)
        x_t = x_t.reshape(B, -1)                ### reshape for easy sampling
        ## Sample z first from S, here we support two types of S, uniform and the same with CTMC forward distribution
        if uniform_sampling: #### sampling with uniform distribution 
            ## For each b, sample a dimension from N1*N2*...*Nk
            sampled_dim = torch.randint(low=0, high=x_t.size(-1), size=(B,), device=x_t.device)
            ## random sample a vector with dimension (B, 1), from C-1 classes. Different from x_t[sampled_dim] 
            idx = torch.arange(B, device=x_t.device)
            ori_class = x_t[idx, sampled_dim]
            new_class = torch.randint(low=0, high=self.num_classes-1, size=(B,), device=x_t.device)
            new_class[new_class>=ori_class] += 1 # make sure new_class != ori_class
            
        else:   #### sampling with forward CTMC distribution 
            m_reshape = m.reshape(B, -1, self.num_classes) # B x D x C
            # step 1: sample a dimension, based on -r_t (x_t | x_t) = beta_t(1 - <x_t, m>)
            m_dot_xt = index_last_dim(m_reshape, x_t) # B x D
            move_out_rate = 1 - m_dot_xt
            changed_dim = torch.multinomial(move_out_rate, num_samples=1).squeeze(-1) # B

            # step 2: for that dimension, sample a new class based on r_t (* | x_t) = beta_t (m - x_t)
            idx = torch.arange(B, device=x_t.device)
            current_value = x_t[idx, changed_dim] # B 
            next_value_prob = m_reshape[idx, changed_dim, :] # B x C
            next_value_prob = set_last_dim(next_value_prob, current_value, value=0) # B x C
            new_class = torch.multinomial(next_value_prob, num_samples=1).squeeze(-1) # B 

        ## Create the new sampled z_t 
        z_t = x_t.clone()
        z_t[idx, changed_dim] = new_class
        z_t = z_t.reshape(x_0.shape)            # B, N1, ..., Nk 

        ## Forward pass of zt
        flogits_zt = denoising_fn(z_t, t)
        flogprob_zt = logits_to_logprob(flogits_zt)
        log_inside_gt_zt = self._log_gt_inner(flogprob_zt, z_t, t, m=m, coef=1.0)

        ## compute q( |x_0) and q(z_t | x_0): (B, N1, ..., Nk, C)
        qt_0_prob = self.qt_0_prob(x_0, t, m=m, return_beta=False)
        qt_0_prob_at_zt = index_last_dim(qt_0_prob, z_t).unsqueeze(-1)
        qt_0_ratio = qt_0_prob / qt_0_prob_at_zt
        set_last_dim(qt_0_ratio, z_t)
        rt_zt_given_anyinput = beta_t * (index_last_dim(m, z_t).unsqueeze(-1)) # B, N1, ..., Nk, 1, this ignores - y

        ## compute nomalizer M 
        if uniform_sampling: 
            M = qt_0_ratio.sum(-1) / D / (self.num_classes-1) # B, N1, ..., Nk
        else:
            ### compute normalizer
            # rt(any_zd | any_zd), B x D x C values 
            moveout_rate_allclass = beta_t_ori * (m - 1) # B, N1, ..., Nk, C
            # rt(yt^d | yt^d), B x D values
            moveout_rate_yt = index_last_dim(moveout_rate_allclass, z_t).unsqueeze(-1) # B, N1, ..., Nk, 1
            normalizer_yt = -moveout_rate_yt.sum(dim=list(range(1, moveout_rate_yt.dim())), keepdim=True) # B, 1, ...,1k,1
            # Z(yt^1:D\d, any_zd),  B x D x C values
            normalizer_any_zd = normalizer_yt + moveout_rate_yt - moveout_rate_allclass # B, N1, ..., Nk, C
            M = (qt_0_ratio * rt_zt_given_anyinput / normalizer_any_zd).sum(-1) # B, N1, ..., Nk
        M = M.sum(dim=list(range(1, M.dim())), keepdim=True)

        ## compute the second term 
        vlb_term2 = -(qt_0_ratio * rt_zt_given_anyinput * log_inside_gt_zt).sum(-1)
        vlb_term2 = vlb_term2 / M
        vlb_loss = vlb_term1 + vlb_term2

        # CE loss
        flogprob_t = logits_to_logprob(flogits_t)
        ce_loss = -index_last_dim(flogprob_t, x_0)

        if conditional_mask is not None:
            assert conditional_mask.shape == x_0.shape
            assert (x_t==x_0)[conditional_mask].all()
            vlb_loss = vlb_loss * (~conditional_mask)
            ce_loss = ce_loss * (~conditional_mask)
            assert vlb_loss.shape == x_0.shape

        vlb_loss = vlb_loss.view(B, -1).sum(-1)
        ce_loss = ce_loss.view(B, -1).sum(-1)
        return vlb_loss.sum(), ce_loss.sum() # add all loss togther 

    # ----------------------------------------------- loss computation combination --------------------------------------------- #
    def compute_loss(self, 
                     logits_t,
                     x_t, 
                     x_0, 
                     t, 
                     m, 
                     coeff_ce=1.,
                     coeff_vlb=1., 
                     conditional_mask=None,
                     denoising_fn=None,
                     simplified_vlb=False):

        if self.num_steps == 0:
            # continuous-time diffusion
            vlb_loss, ce_loss = self.continuous_time_loss(logits_t, x_t, x_0, t, m, conditional_mask, denoising_fn, uniform_sampling=False, simplified_vlb=simplified_vlb)
        else:
            # discrete-time diffusion
            vlb_loss, ce_loss = self.discrete_time_loss(logits_t, x_t, x_0, t, m, conditional_mask, simplified_vlb=simplified_vlb)

        loss = coeff_vlb * vlb_loss + coeff_ce * ce_loss

        output_dict = {'loss'    : loss,
                       'vlb_loss': vlb_loss,
                       'ce_loss' : ce_loss,} 
        return output_dict
    
    # -------------------------------------------------- MCMC computation   ---------------------------------------------------- #
    def mcmc_corrector(self, denoising_fn, x_t, t, step_size, max_steps=10, min_stay_prob=0.2, m=None, conditional_mask=None):
        """
        denoising_fn: (B, N1, ..., Nk, C) -> (B, N1, ..., Nk, C)
        x_t        : (B, N1, ..., Nk)
        t          : (B,)
        step_size  : (B,)  (depends on t)
        conditional_mask : (B, N1, ..., Nk) or None
        """
        # TODO: think step size as a function of t.  
        z_n = x_t
        if type(step_size) == float:
            B = t.shape[0]
            step_size = torch.full((B,), step_size, dtype=torch.float32).to(x_t.device)
        for n in range(max_steps):
            fprob_t = logits_to_prob(denoising_fn(x_t, t)) # TODO: add temperature later
            z_n, step_size = self._mcmc_step(fprob_t, t, z_n, step_size, min_stay_prob=min_stay_prob, m=m, conditional_mask=conditional_mask)
            # TODO: consider reduce step_size adaptively 
        return z_n
    
    def _mcmc_step(self, fprob_t, t, z_n, delta_n, min_stay_prob=0.2, m=None, conditional_mask=None):
        """
        fprob_t: (B, N1, ..., Nk, C)
        t      : (B,)
        z_n    : (B, N1, ..., Nk)
        delta_n: (B,)
        
        conditional_mask : (B, N1, ..., Nk) or None
        """
        # compute unnormailzed prob
        prob_unnorm, beta_t, _ = self._gt_inner(fprob_t, z_n, t, m=m, coef=2)    # (B, N1, ..., Nk, C)
        prob_unnorm = (beta_t * delta_n).view([-1]+[1]*z_n.dim()) * prob_unnorm 
        max_scale = prob_unnorm.sum(dim=-1).flatten(start_dim=1).max(-1)[0]      # (B,)

        # adjust the scale of prob_unnorm to prevent large delta_n 
        for i, scale in enumerate(max_scale):
            if scale >= (1-min_stay_prob):
                prob_unnorm[i] = prob_unnorm[i] / scale * (1-min_stay_prob)
                delta_n[i] = delta_n[i] * (1-min_stay_prob) / scale

        # compute the prob 
        broadcast_idx = get_broadcast_idx(z_n.shape)
        prob_unnorm[broadcast_idx + [z_n]] += 1 - prob_unnorm.sum(-1) 
        prob = prob_unnorm 

        z_n_new = sample_categorical(prob)
        if conditional_mask is not None:
            assert conditional_mask.shape == z_n.shape
            z_n_new[conditional_mask] = z_n[conditional_mask]

        # sampling from this distribution
        return z_n_new, delta_n

    # -------------------------------------------------- Backward sampling  ---------------------------------------------------- #
    def sample_step(self, denoising_fn, x_t, t, s, m=None, mcmc_num_steps=0, mcmc_step_size=None, conditional_mask=None):
        """ From time step t to time step s  (t>s)
        denoising_fn : (B, N1, ..., Nk) -> (B, N1, ..., Nk, C)
        x_t         : (B, N1, ..., Nk)
        t           : (B,)
        s           : (B,)
        m           : (B, N1, ..., Nk, C) or None or (C)
        mc_step_size: (B,) or None
        conditional_mask : (B, N1, ..., Nk) or None
        """
        # compute fprob_t 
        fprob_t = logits_to_prob(denoising_fn(x_t, t))           # (B, N1, ..., Nk, C)
        # compute P(x_s | x_t)
        prob_s = self.ps_t_prob(fprob_t, x_t, t=t, s=s, m=m)    # (B, N1, ..., Nk, C)
        # for s = 0, change prob_s to fprob_t (final step sampling towards x0)
        prob_s[s==0] = fprob_t[s==0]

        # sample x_s 
        x_s = sample_categorical(prob_s)                        # (B, N1, ..., Nk)
        if conditional_mask is not None:
            assert conditional_mask.shape == x_s.shape
            x_s[conditional_mask] = x_t[conditional_mask]
        
        if mcmc_num_steps >0 and self.num_steps == 0:
           assert mcmc_step_size is not None
           x_s = self.mcmc_corrector(denoising_fn, x_s, s, mcmc_step_size, max_steps=mcmc_num_steps, min_stay_prob=0.2, m=m, conditional_mask=conditional_mask)
        return x_s
