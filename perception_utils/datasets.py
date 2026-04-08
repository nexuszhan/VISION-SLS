import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

class CustomDataset(Dataset):
    def __init__(self, data_path, data_path_dino, device, z_dim, init_z=None):
        # _, _, self.data_x = torch.load(data_path) # states
        if data_path_dino is None:
            self.data_x, self.data_z = torch.load(data_path)  # states
        else:
            _, self.data_x = torch.load(data_path)  # states
            self.data_z = torch.load(data_path_dino) # dino labels

    def __len__(self):
        return len(self.data_z)

    def __getitem__(self, index):
        return self.data_z[index], self.data_x[index]

class CustomDatasetDinoErr(Dataset):
    def __init__(self, data_path, err, device, z_dim, init_z=None):
        # _, self.data_x = torch.load(data_path)  # states
        self.data_x, _ = torch.load(data_path)  # states
        self.data_err = err

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_err[index]

class CustomDatasetErr(Dataset):
    def __init__(self, data_path, device, z_dim, init_z=None):
        self.data_x, self.data_u, self.data_err = torch.load(data_path) # states
        self.data_xu = torch.cat((self.data_x, self.data_u), dim=1)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_xu[index], self.data_err[index]
class CustomDatasetCT(Dataset):
    def __init__(self, data_path, device, z_dim, N_nodes=1, init_z=None):
        self.data_y, self.data_u, self.data_traj, self.t = torch.load(data_path)
        self.data_y = self.data_y.to(device)
        self.data_u = self.data_u.to(device)
        self.data_traj = self.data_traj.to(device)
        if init_z is None:
            temp = 2*0.0*torch.rand(self.data_y.shape[0], N_nodes, z_dim, device=device).requires_grad_(True) - 0.0
            self.init_z = nn.Parameter(temp)
        else:
            self.init_z = init_z.to(device=device)

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, index):
        return self.data_y[index], self.data_u[index], self.init_z[index], self.data_traj[index]

class CustomDatasetReconstruction(Dataset):
    def __init__(self, data_path):
        self.data_y, _ = torch.load(data_path)

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, index):
        return self.data_y[index]