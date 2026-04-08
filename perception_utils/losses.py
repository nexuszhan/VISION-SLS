import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

def compute_cpc_loss(z, z_pos, z_next, multiplier=1., thresh=0.1):
    # z_next is the computed next state, z_pos is the next state label, z is the previous state
    bs = z.shape[0]

    neg_dot_products = torch.mm(z_next, z.t()) # b x b
    neg_dists = -((z_next ** 2).sum(1).unsqueeze(1) - 2* neg_dot_products + (z ** 2).sum(1).unsqueeze(0))
    idxs = np.arange(bs)
    # Set to minus infinity entries when comparing z with z - will be zero when apply softmax
    neg_dists[idxs, idxs] = float('-inf') # b x b+1

    pos_dot_products = (z_pos * z_next).sum(dim=1) # b
    pos_dists = -((z_pos ** 2).sum(1) - 2* pos_dot_products + (z_next ** 2).sum(1))
    pos_dists = multiplier * pos_dists.unsqueeze(1) # b x 1

    dists = torch.cat((neg_dists, pos_dists), dim=1) # b x b+1
    dists = F.log_softmax(dists, dim=1) # b x b+1
    loss = -dists[:, -1].mean() # Get last column with is the true pos sample

    # Gaussian loss
    # gauss = 3*torch.log(torch.exp(thresh * neg_dists ** 2).mean())
    # gauss = (-1. * torch.pdist(z_next).pow(2.)).exp().mean().log()
    return loss# + gauss

def reconstruction_loss(r, y_orig):
    # l2_loss = (r - y_orig).reshape(-1, y_orig.shape[-2], y_orig.shape[-1]).norm(dim=[1,2]).mean()
    # weight_rebal = torch.ones_like(y_orig).to(y_orig.device) / 10.0 + (1.0 - 1.0 / 10.0) * y_orig
    # weight_rebal = weight_rebal.requires_grad_(False)
    # loss = nn.functional.binary_cross_entropy(r, y_orig, weight=weight_rebal)
    loss = nn.functional.binary_cross_entropy(r, y_orig)
    return loss

def single_step_dyn_loss(ztp1, ztp1_label):
    return (ztp1_label - ztp1).norm(dim=1).mean()

def multi_step_dyn_loss(r, y_orig):
    return (r - y_orig).reshape(-1, y_orig.shape[-2], y_orig.shape[-1]).norm(dim=[1,2]).mean()
