import torch.autograd.functional
from torch.func import jacrev
from omegaconf import DictConfig, OmegaConf
from perception_utils.losses import *
from perception_utils.networks import *
from perception_utils.datasets import *
from perception_utils.utils import *
from perception_utils.train import train_network
# import hydra
import os

base_path = os.path.dirname(__file__)
# mass, grav, length, damp = 1., 9.81, 5., 0.1
# mgl = mass * grav / length
# nx, nu, ny = 4, 2, 3
def dyn_x(x):
    return torch.hstack([x[:,3:4] * torch.cos(x[:,2:3]),
                      x[:,3:4] * torch.sin(x[:,2:3]),
                      x[:,4:5]])

def loss_gramian(net, x):
    Ls = []
    Os = []
    C_norm = net.C / net.C.norm(dim=1, keepdim=True)
    O1 = C_norm[None,:].repeat(x.shape[0], 1, 1)
    def O2_func(x):
        return (O1 @ dyn_x(x)[:,:,None]).reshape(x.shape[0], -1)

    # O2 = torch.einsum("bibj->bij", torch.autograd.functional.jacobian(O2_func, x))
    O2 = torch.einsum("bibj->bij", jacrev(O2_func)(x))

    def O3_func(x):
        return (O2 @ dyn_x(x)[:,:,None]).reshape(x.shape[0], -1)
    # O3 = torch.einsum("bibj->bij", torch.autograd.functional.jacobian(O3_func, x))
    O3 = torch.einsum("bibj->bij", jacrev(O3_func)(x))

    O_gram = torch.cat([O1, O2, O3], dim=1)
    U, S, Vh = torch.linalg.svd(O_gram, full_matrices=True)
    loss_gram = -(S.abs().min(dim=1)[0].clamp(max=1.)).mean()

    return loss_gram

def loss_uni(net, z_dino, x, params=None, epoch=None):
    y_red_pred = net(z_dino)
    # y_red_pred = y_red_pred / y_red_pred.norm(dim = 1, keepdim = True)
    # loss_gramian_val = loss_gramian(net, x)
    C_norm = net.C / net.C.norm(dim=1, keepdim=True)
    loss = ((C_norm @ x.T).T - y_red_pred).norm(dim = 1).mean() #+ loss_gramian_val + 0.00*torch.norm(C_norm, p=1)

    return loss

def main():
    params = OmegaConf.load("config/learn_model.yaml")
    if torch.has_cuda and params.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # load dataset
    data = CustomDataset(os.path.join(base_path, params.dataset.load_filename),
                         None,#os.path.join(base_path, params.dataset.load_dino_filename),
                         device, params.model.z_dim)
    # data.data_x = torch.hstack([data.data_x, torch.randn(data.data_x.shape[0], 1).to(data.data_x.device)])
    
    data.data_x = data.data_x.to(device).float()
    data.data_z = data.data_z.to(device).float()
    # generate model
    model = SupervisedDinoObservability(params.model).to(device)
    # checkpoint = torch.load(os.path.join(base_path, params.train.load_ckpt), map_location=device)
    # model.load_state_dict(checkpoint)
    # define loss function
    loss_fn = loss_uni
    # run train loop
    train_network(model, loss_fn, data, params)

if __name__ == '__main__':
    main()
