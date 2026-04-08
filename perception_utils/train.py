from perception_utils.losses import *
from perception_utils.networks import *
from perception_utils.datasets import *
from perception_utils.utils import *
# import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

import os
base_path = os.path.dirname(__file__)

def train_network(net, loss_fn, dataset, params):
    """
    Train a network given a dataset and optimization parameters.
    """
    # if params.train.wandb.enabled:
    #     wandb.init(
    #         project=params.train.wandb.project, entity=params.train.wandb.entity
    #     )
    # OmegaConf.save(config=params, f=os.path.join(wandb.run.dir, '../../../run_config.yaml'))
    # text_file = open(os.path.join(wandb.run.dir, '../../../network_params.txt'), "w")
    # n = text_file.write(str(net))
    # text_file.close()

    net.train()
    # optimizer = optim.Adam(net.parameters(), params.train.lr)
    optimizer = optim.Adam(list(net.parameters()), params.train.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, params.train.epochs
    )
    # scheduler_fixed = optim.lr_scheduler.StepLR(optimizer, 5000, gamma=0.95)
    # scheduler = optim.lr_scheduler.ChainedScheduler([scheduler_cosine, scheduler_fixed])

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, params.dataset.split
    )
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.train.batch_size
    )
    data_loader_eval = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset)
    )

    loss_lst = np.zeros(params.train.epochs)

    best_loss_train = np.inf
    best_loss_val = np.inf
    
    with tqdm(total=params.train.epochs, desc="Epoch") as pbar:
        for epoch in range(params.train.epochs):
            loss_curr = []
            for y_batch, x_batch in data_loader_train:
                loss = loss_fn(
                    net, y_batch, x_batch, params=params, epoch=epoch
                )
                loss_curr.append(dcn(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            loss_train_mean = np.array(loss_curr).mean()
            with torch.no_grad():
                losses_eval = []
                for y_all, x_all in data_loader_eval:
                    loss_eval = loss_fn(
                        net, y_all, x_all, params=params, epoch=epoch
                    )
                    losses_eval.append(dcn(loss_eval))
                loss_lst[epoch] = np.array(losses_eval).mean()
                if params.train.wandb.enabled:
                    # wandb.log({"val loss": loss_lst[epoch]}, step=epoch)
                    # wandb.log({"train loss": loss_train_mean}, step=epoch)
                    pbar.set_postfix(val_loss=loss_lst[epoch], train_loss=loss_train_mean)
                    pbar.update(1)
                if params.saving.save_best_model is not None and loss_lst[epoch] < best_loss_val:
                    # torch.save(net.state_dict(), os.path.join(base_path, params.train.save_ckpt))
                    # torch.save(net.state_dict(), wandb.run.dir + '/../../..' + params.train.save_ckpt_val)
                    torch.save(net.state_dict(), params.train.save_ckpt_val)
                    best_loss_val = loss_lst[epoch]
                if params.saving.save_best_model is not None and loss_train_mean < best_loss_train:
                    # torch.save(net.state_dict(), os.path.join(base_path, params.train.save_ckpt))
                    # torch.save(net.state_dict(), wandb.run.dir + '/../../..' + params.train.save_ckpt_train)
                    torch.save(net.state_dict(), params.train.save_ckpt_train)
                    best_loss_train = loss_train_mean
            # print("Epoch: {}, Loss: {}, Loss_eval: {}".format(epoch, loss_train_mean, loss_lst[epoch]))

    return loss_lst

def train_network_descend(net, u_net, loss_fn, params):
    """
    Train a network given a dataset and optimization parameters.
    """
    # if params.train.wandb.enabled:
    #     wandb.init(
    #         project=params.train.wandb.project, entity=params.train.wandb.entity
    #     )
    # net.train()
    # optimizer = optim.Adam(net.parameters(), params.train.lr)
    optimizer = optim.Adam([u_net], params.train.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, params.train.epochs
    )

    loss_lst = np.zeros(params.train.epochs)

    best_loss = np.inf

    for epoch in tqdm(range(params.train.epochs)):
    # for epoch in range(params.train.epochs):
        loss_curr = []
        loss = loss_fn(
            net, u_net, params=params
        )
        loss_curr.append(dcn(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_train_mean = np.array(loss_curr).mean()
        print(loss_train_mean)
        # with torch.no_grad():
        #     losses_eval = []
        #     for y_all, x_all in data_loader_eval:
        #         loss_eval = loss_fn(
        #             net, y_all, x_all, params=params
        #         )
        #         losses_eval.append(dcn(loss_eval))
        #     loss_lst[epoch] = np.array(losses_eval).mean()
        #     if params.train.wandb.enabled:
        #         wandb.log({"val loss": loss_lst[epoch]}, step=epoch)
        #         wandb.log({"train loss": loss_train_mean}, step=epoch)
        #     if params.saving.save_best_model is not None and loss_eval.item() < best_loss:
        #         torch.save(net.state_dict(), os.path.join(base_path, params.train.save_ckpt))
        #         best_loss = loss_lst[epoch]
        # print("Epoch: {}, Loss: {}, Loss_eval: {}".format(epoch, loss_train_mean, loss_lst[epoch]))

    return loss_lst