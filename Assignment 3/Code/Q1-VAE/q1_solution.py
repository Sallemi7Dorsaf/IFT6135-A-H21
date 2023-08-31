"""
Solutions for Question 2 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    # log_likelihood_bernoulli
    # TODO
    p = target*torch.log(mu)
    q = (1-target)*torch.log(1-mu)
    return (p+q).sum(1)
    #raise NotImplementedError("Implement as part of assignment 3.")


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    # log normal
    # TODO
    #scale = torch.exp(logvar)
    scale = torch.stack([torch.diag(i) for i in torch.exp(logvar)])
    dist = torch.distributions.MultivariateNormal(mu, scale)
    log_p = dist.log_prob(z)
    return log_p

    #raise NotImplementedError("Implement as part of assignment 3.")


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size)
    - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    # log_mean_exp
    # TODO
    m, idx = torch.max(y, dim=1)
    ex = torch.exp(y-m.unsqueeze(1))
    return (m + torch.log(ex.mean(dim=1)))
    #raise NotImplementedError("Implement as part of assignment 3.")


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # kld
    # TODO
    qdist = torch.distributions.MultivariateNormal(mu_q, torch.stack([torch.diag(i) for i in torch.exp(logvar_q)]))
    pdist = torch.distributions.MultivariateNormal(mu_p, torch.stack([torch.diag(i) for i in torch.exp(logvar_p)]))
    #return torch.nn.functional.kl_div(qdist, pdist)
    return torch.distributions.kl.kl_divergence(qdist, pdist)
    #raise NotImplementedError("Implement as part of assignment 3.")


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    # kld
    # TODO
    qdist = torch.distributions.MultivariateNormal(mu_q.mean(dim=1),torch.stack([torch.diag(i) for i in torch.exp(logvar_q.mean(dim=1))]))
    pdist = torch.distributions.MultivariateNormal(mu_p.mean(dim=1), torch.stack([torch.diag(i) for i in torch.exp(logvar_p.mean(dim=1))]))
    #print('1')
    #z = qdist.rsample((num_samples,))
    #print('2')
    #log_qzx = qdist.log_prob(z)
    #log_pz = pdist.log_prob(z)
    #kl = (log_qzx - log_pz)
    #print('3')
    #return kl.sum(-1)
    
    # kld
    return torch.distributions.kl.kl_divergence(qdist, pdist)
    #raise NotImplementedError("Implement as part of assignment 3.")
