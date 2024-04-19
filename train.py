from typing import Optional
import pathlib
import argparse

import numpy as np

import src.models as models
import src.datasets as datasets
import src.losses as losses
import src.utils as utils

import torch

def train(contact_problem: str,
          log_path: pathlib.Path,
          num_workers: int = 4,
          learning_rate: float = 1e-3, 
          epochs: int = 100,
          epochs_per_test: int = 1,
          epochs_early_stop: Optional[int] = None,
          tensorboard_log: bool = False):
    """Train a contact deformation model for a given contact interaction, 
       logging trained model checkpoints and evaluation losses.
 
    Keyword arguments:
    
    contact_problem:   Name of the contact problem to be trained.

    log_path:          Path to store the logging data of the different trainings.
                       Model checkpoints (.pth) and tensorboard log are stored in this path.

    num_workers:       Number of worker processes for data loading.
                       May require adjustments for optimal results in different datasets & systems.

    learning_rate:     Initial learning rate used during training. Default value of 1e-3.

    epochs:            Number of training epochs. Maximum epochs in case of using early stoping.

    epochs_per_test:   Number of consecutive training epochs required for a test error evaluation.

    epochs_early_stop: Number of consecutive non improving epoch for early stoping. 
                       If not provided, early stoping is not used. 

    tensorboard_log:   Record a tensorboard log with the train & test losses per epoch.
    """

    seed = 0
    torch.manual_seed(seed)
    print("Torch Seed: ", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch Device: ", device)

    ######

    if contact_problem == "jelly3D_thingi3D":

        path_data     = pathlib.Path("./data/contacts/jelly3D_thingi3D/")
        path_object   = pathlib.Path("./data/objects/jelly3D/")
        
        path_collider = pathlib.Path("./data/colliders/thingi3D/")
        dataset_shapes = datasets.ShapeDataset(path_collider)

        pattern = utils.getPattern(path_collider.parents[1])
        model = models.DefModel(path_object,
                        dataset_shapes,
                        patch_pattern = pattern,
                        grid_size_sdf = 50,
                        grid_size_descriptor = 0,
                        mode_frame = models.ModeFrame.RANDOM_TANGENT,
                        biased_frame = False,
                        samples_per_frame = 50,
                        frame_radius = 0.1,
                        patch_radius = 0.2,
                        max_distance = 0.15,
                        mask_factor = 0.3)

        invariance_weight = 0.2
        
        samples_grid_ids = torch.arange(52*5*7*3*5).reshape(52,5,7,3,5)

        data_slice_A = samples_grid_ids[0:30, :, :, :, :].reshape(-1)
        data_slice_A = data_slice_A[torch.randperm(data_slice_A.shape[0])]

        data_slice_B = samples_grid_ids[30:52, :, :, :, :].reshape(-1)
        data_slice_B = data_slice_B[torch.randperm(data_slice_B.shape[0])]

        data_slice_train = data_slice_A
        data_slice_test  = data_slice_B

    else:
        
        print("contact_problem does not exist!")

    model.to(device)

    loader_params_train = {'batch_size': 1024,
                           'shuffle': True,
                           'num_workers': num_workers}
    loader_params_test = {'batch_size': 5024,
                          'shuffle': True,
                          'num_workers': num_workers}

    batch_nodes_train = 2
    batch_nodes_test = 2

    dataset_train = datasets.ContactDataset(device, 
                                            model, 
                                            path_object, 
                                            path_data, 
                                            data_slice_train, 
                                            near_nodes = True, 
                                            batch_nodes = batch_nodes_train)

    dataset_test = datasets.ContactDataset(device, 
                                           model, 
                                           path_object, 
                                           path_data, 
                                           data_slice_test, 
                                           near_nodes = True, 
                                           batch_nodes = batch_nodes_test)
    
    dataset_train_loader = torch.utils.data.DataLoader(dataset_train, **loader_params_train)
    dataset_test_loader = torch.utils.data.DataLoader(dataset_test, **loader_params_test)

    # Training
    #############################################################

    loss = losses.MSENLoss()
    loss.to(device)
    loss_invariance = losses.MSENLoss()
    loss_invariance.to(device)

    optimizer = torch.optim.Adam(model.collider_net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.9)
    
    if epochs_early_stop is not None:
        stoper = losses.EarlyStopper(patience=epochs_early_stop, min_delta=0)
    
    checkpoint_path = log_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if tensorboard_log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_path.absolute()))

    # Normalize Network

    model.train()

    model.collider_net.clearMean()

    for (q, z, s, x, x_target) in dataset_train_loader:

        q = q.to(device)
        z = z.to(device)
        x = x.to(device)
        x_target = x_target.to(device)

        q_patch_relative, patch_shape, x_corr_local = model.getLocalTransform(q, z, s, x, x_target)
        model.collider_net.accumMean(q_patch_relative, patch_shape, x_corr_local)

    model.collider_net.divideMean(len(dataset_train_loader))

    model.collider_net.clearStd()

    for (q, z, s, x, x_target) in dataset_train_loader:

        q = q.to(device)
        z = z.to(device)
        x = x.to(device)
        x_target = x_target.to(device)

        q_patch_relative, patch_shape, x_corr_local = model.getLocalTransform(q, z, s, x, x_target)
        model.collider_net.accumStd(q_patch_relative, patch_shape, x_corr_local)

    model.collider_net.divideStd(len(dataset_train_loader))

    # Normalize Loss

    model.train()

    loss.clearNormalization()
    loss_invariance.clearNormalization()

    for (q, z, s, x, x_target) in dataset_train_loader:

        q = q.to(device)
        z = z.to(device)
        x = x.to(device)
        x_target = x_target.to(device)

        r = model(q, z, s, x)

        r_diffs = r - torch.mean(r, dim=1)[:,None,:]

        loss.accumNormalization(r, x_target)
        loss_invariance.accumNormalization(r_diffs, 0.0*x_target)

    loss.divideNormalization(len(dataset_train_loader))
    loss_invariance.divideNormalization(len(dataset_train_loader))

    print("Loss mse_loss_norm: ", loss.mse_loss_norm)
    print("Loss invariance mse_loss_norm: ", loss_invariance.mse_loss_norm)

    print(0, "Loss Train: ", 1.0)
    
    if tensorboard_log:
        writer.add_scalar('Loss Train', 1.0, 0)
        writer.add_scalar('Loss Train Target', 1.0, 0)
        writer.add_scalar('Loss Train Invariance', 1.0, 0)

    # Train loop

    model.train()

    loss_test_best = np.inf

    for i in range(epochs):

        print("Saving checkpoint epoch " + str(i))
        filename = checkpoint_path / f"checkpoint-{i}.pth"
        torch.save(model.collider_net.state_dict(), filename)

        if i % epochs_per_test == 0:
            
            loss_test_target = 0.0
            loss_test_invariance = 0.0

            loss_test = 0.0

            with torch.no_grad():

                for (q, z, s, x, x_target) in dataset_test_loader:

                    q = q.to(device)
                    z = z.to(device)
                    x = x.to(device)
                    x_target = x_target.to(device)

                    r = model(q, z, s, x)

                    r_diffs = r - torch.mean(r, dim=1)[:,None,:]

                    lo = loss(r, x_target) / len(dataset_test_loader)
                    lor = loss_invariance(r_diffs, 0.0*x_target) / len(dataset_test_loader)

                    l = (1.0-invariance_weight)*lo + invariance_weight*lor

                    loss_test_target += lo.cpu().detach().item()
                    loss_test_invariance += lor.cpu().detach().item()
                    loss_test += l.cpu().detach().item()

                print(i, "Loss Test: ", loss_test)

                if tensorboard_log:
                    writer.add_scalar('Loss Test', loss_test, i)
                    writer.add_scalar('Loss Test Target', loss_test_target, i)
                    writer.add_scalar('Loss Test Invariance', loss_test_invariance, i)

            if loss_test <= loss_test_best:
                filename = log_path / "model.pth"
                torch.save(model.collider_net.state_dict(), filename)
                
                loss_test_best = loss_test

            if epochs_early_stop is not None and stoper.step(loss_test, epochs_per_test):
                break

        loss_train_target = 0.0
        loss_train_invariance = 0.0

        loss_train = 0.0

        for (q, z, s, x, x_target) in dataset_train_loader:

            q = q.to(device)
            z = z.to(device)
            x = x.to(device)
            x_target = x_target.to(device)

            optimizer.zero_grad()

            r = model(q, z, s, x)

            r_diffs = r - torch.mean(r, dim=1)[:,None,:]

            lo = loss(r, x_target) / len(dataset_train_loader)
            lor = loss_invariance(r_diffs, 0.0*x_target) / len(dataset_train_loader)

            l = (1.0-invariance_weight)*lo + invariance_weight*lor

            l.backward()
            optimizer.step()

            loss_train_target += lo.cpu().detach().item()
            loss_train_invariance += lor.cpu().detach().item()
            loss_train += l.cpu().detach().item()

        print(i+1, "Loss Train: ", loss_train)

        if tensorboard_log:
            writer.add_scalar('Loss Train', loss_train, i+1)
            writer.add_scalar('Loss Train Target', loss_train_target, i+1)
            writer.add_scalar('Loss Train Invariance', loss_train_invariance, i+1)

        scheduler.step(loss_train)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--contact_problem', type = str, required=True,
                        help='Name of the contact problem to be trained')

    parser.add_argument('--log_path', type = pathlib.Path, required=True,
                        help='Path to store the logging data of the different trainings. \
                              Model checkpoints (.pth) and tensorboard log are stored in this path')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading. \
                              May require adjustments for optimal results in different datasets & systems')

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate used during training. Default value of 1e-3')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs. Maximum epochs in case of using early stoping')
    
    parser.add_argument('--epochs_per_test', type=int, default=1,
                        help='Number of consecutive training epochs required for a test error evaluation')
    
    parser.add_argument('--epochs_early_stop', type=int,
                        help='Number of consecutive non improving epoch for early stoping')
    
    parser.add_argument('--tensorboard_log', action='store_true',
                        help='Record a tensorboard log with the train & test losses per epoch')
    
    args = parser.parse_args()

    train(**vars(args))