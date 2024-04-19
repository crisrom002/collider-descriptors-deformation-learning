from typing import Optional
import pathlib
import argparse

import numpy as np
import torch

import src.datasets as datasets
import src.models as models
import src.utils as utils

from scipy.spatial.transform import Rotation as R

import polyscope as ps
import polyscope.imgui as psim

import time
import struct

def evaluate(contact_problem: str,
             model_file: Optional[pathlib.Path] = None,
             precompute_descriptor: bool = False,
             output_path: Optional[pathlib.Path] = None,
             show_near_nodes: bool = False):
    """Evaluate the data and results for a given contact interaction and trained model. 
       If no model is provided, the different ground truth contact deformations can be explored.
       Optionally, the training process can be evaluated provided the directory with model checkpoints.
       
    Keyword arguments:
    
    contact_problem:       Name of the contact problem to be evaluated.

    model_file:            Path to trained model weights file (.pth).
                           Optionally, path to a log directory generated during training.
                           If nothing is provided, model weights are randomly initialized.

    precompute_descriptor: Optional grid precomputation of the collider descritor.

    output_path:           Optional path to the desired output directory. If provided, different 
                           vertex position sequence files (.pc2) are recorded for the collider (collider.pc2), 
                           reduced deformation (reduced.pc2), full ground truth (full.pc2) and model (model.pc2) results.

    show_near_nodes:       Optional flag to precompute and visualize the nodes near and inside the collider. 
                           Used for debuging the subset of nodes in the supervised training.      
    """

    seed = 0
    torch.manual_seed(seed)
    print("Torch Seed: ", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch Device: ", device)

    ######

    if contact_problem == "jelly3D_sphereSpiky3D":

        path_data     = pathlib.Path("./data/contacts/jelly3D_sphereSpiky3D/")
        path_object   = pathlib.Path("./data/objects/jelly3D_script/")

        path_collider = pathlib.Path("./data/colliders/sphereSpiky3D/")
        dataset_shapes = datasets.ShapeDataset(path_collider)

        pattern = utils.getPattern(path_collider.parents[1])
        model = models.DefModel(path_object,
                                dataset_shapes,
                                pattern,
                                grid_size_sdf = 50,
                                grid_size_descriptor = 50 if precompute_descriptor else 0,
                                mode_frame = models.ModeFrame.RANDOM_TANGENT,
                                biased_frame = True,
                                samples_per_frame = 50,
                                frame_radius = 0.1,
                                patch_radius = 0.2,
                                max_distance = 0.15,
                                mask_factor = 0.3)
            
        samples_grid = np.array([1,835,1,1,1])

    elif contact_problem == "jelly3D_thingi3D":

        path_data     = pathlib.Path("./data/contacts/jelly3D_thingi3D/")
        path_object   = pathlib.Path("./data/objects/jelly3D/")

        path_collider = pathlib.Path("./data/colliders/thingi3D/")
        dataset_shapes = datasets.ShapeDataset(path_collider)

        pattern = utils.getPattern(path_collider.parents[1])
        model = models.DefModel(path_object,
                                dataset_shapes,
                                patch_pattern = pattern,
                                grid_size_sdf = 50,
                                grid_size_descriptor = 50 if precompute_descriptor else 0,
                                mode_frame = models.ModeFrame.RANDOM_TANGENT,
                                biased_frame = True,
                                samples_per_frame = 50,
                                frame_radius = 0.1,
                                patch_radius = 0.2,
                                max_distance = 0.15,
                                mask_factor = 0.3)
        
        samples_grid = np.array([52,5,7,3,5])

    else:

        print("contact_problem does not exist!")

    model.to(device)
    model.eval()

    num_samples = np.prod(samples_grid)
    data_slice = torch.arange(num_samples)

    dataset = datasets.ContactDataset(device,
                                      model,
                                      path_object, 
                                      path_data, 
                                      data_slice, 
                                      near_nodes = show_near_nodes, 
                                      batch_nodes = 0)
    dataset.near_nodes = False

    num_model_checkpoints = 0

    if model_file is not None:

        if model_file.is_file():
            state_dict_save = torch.load(model_file, map_location=device)
            model.collider_net.load_state_dict(state_dict_save)

        elif model_file.is_dir():
            checkpoint_path = model_file / "checkpoints"
            num_model_checkpoints = len(list(checkpoint_path.iterdir()))

    # Recording
    #############################################################

    def save_pc2(filename, data):

        data = data.astype(np.float32)
        num_samples = data.shape[0]
        num_vertices = data.shape[1]

        filename.parents[0].mkdir(parents=True, exist_ok=True)

        with filename.open("wb") as f:
            header_format = '<12siiffi'
            header = struct.pack(header_format, b'POINTCACHE2\0', 1, num_vertices, 0, 1, num_samples)
            f.write(header)

            f.write(data.tobytes())

        print("Saved pc2:", filename)

    if output_path is not None:

        data_reduced = []
        data_full = []
        data_model = []
        data_collider = []

        for sample in range(num_samples):

            print("computing sample... ", sample, " / ", num_samples , end='\r')

            q, z, s, x, x_target = dataset[sample]

            q = q.to(device)
            z = z.to(device)
            x = x.to(device)

            x_model = model(q, z, s, x)[0].cpu().detach().numpy()

            data_reduced.append(x.cpu().detach().numpy())
            data_full.append(x_target.cpu().detach().numpy())
            data_model.append(x_model)

            #
            V_collider, _ = dataset_shapes[s]      
            W_collider = torch.cat((V_collider, torch.ones((V_collider.shape[0],1))), dim=1)
            x_collider = torch.matmul(W_collider, z.cpu()).cpu().detach().numpy()
            data_collider.append(x_collider)

        data_reduced = np.array(data_reduced)
        data_full = np.array(data_full)
        data_model = np.array(data_model)
        data_collider = np.array(data_collider)

        save_pc2(output_path / "reduced.pc2", data_reduced)
        save_pc2(output_path / "full.pc2", data_full)
        save_pc2(output_path / "model.pc2", data_model)
        save_pc2(output_path / "collider.pc2", data_collider)

        return
     
    # Visualization
    #############################################################

    ps.init()

    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_SSAA_factor(4)

    sample = 0
    sample_old = -1

    show_model = False

    model_num = num_model_checkpoints-1
    model_num_old = -1

    play = False
    sleep = False

    custom_collider = False
    pos = np.array([0.0, 0.0, 0.0])
    rot_axis = np.array([0.0, 0.0, 0.0])
    shape = 0

    s_old = -1
    W_collider = None

    eval_time = 0.0

    ###########

    collider_color = (1.0,1.0,1.0)
    reduced_color = (0.5,1.0,0.1)
    full_color = (1.0,0.5,0.1)
    model_color = (1.0,24.0/255.0,12.0/255.0)

    V_collider, F_collider = dataset_shapes[0]
    F_collider = F_collider.cpu().detach().numpy()

    _, _, _, _, V_object = dataset[0]
    F_object = np.load(path_object / "info" / "facetIds.npy")

    collider_mesh = ps.register_surface_mesh("collider", V_collider, F_collider, smooth_shade=False)
    collider_mesh.set_color(collider_color)
    collider_mesh.set_transparency(0.75)
    collider_mesh.set_back_face_policy("cull")
    ps.remove_surface_mesh("collider")
    collider_mesh = None

    reduced_mesh = ps.register_surface_mesh("reduced", V_object, F_object, smooth_shade=True)
    reduced_mesh.set_color(reduced_color)
    ps.remove_surface_mesh("reduced")
    reduced_mesh = None

    full_mesh = ps.register_surface_mesh("full", V_object, F_object, smooth_shade=True)
    full_mesh.set_color(full_color)
    ps.remove_surface_mesh("full")
    full_mesh = None

    model_mesh = ps.register_surface_mesh("model", V_object, F_object, smooth_shade=True)
    model_mesh.set_color(model_color)
    ps.remove_surface_mesh("model")
    model_mesh = None 

    reduced_nodes = ps.register_point_cloud("reduced nodes", V_object)
    reduced_nodes.set_color(reduced_color)
    ps.remove_point_cloud("reduced nodes")
    reduced_nodes = None

    full_nodes = ps.register_point_cloud("full nodes", V_object)
    full_nodes.set_color(full_color)
    ps.remove_point_cloud("full nodes")
    full_nodes = None

    model_nodes = ps.register_point_cloud("model nodes", V_object)
    model_nodes.set_color(model_color)
    ps.remove_point_cloud("model nodes")
    model_nodes = None

    def main_loop():

        nonlocal dataset, model, show_model, sample, sample_old, model_num, model_num_old, \
                 play, sleep, custom_collider, pos, rot_axis, shape, s_old, W_collider, eval_time, \
                 F_object, collider_mesh, reduced_mesh, full_mesh, model_mesh, \
                 reduced_nodes, full_nodes, model_nodes

        _, sample = psim.SliderInt("Sample", sample, 0, num_samples-1)

        if samples_grid[0] > 1:
            if psim.Button("+ shape"):
                sample = sample + np.prod(samples_grid[1:])

            psim.SameLine()

            if psim.Button("- shape"):
                sample = sample - np.prod(samples_grid[1:])

        if samples_grid[1] > 1:
            if psim.Button("+ state"):
                sample = sample + np.prod(samples_grid[2:])

            psim.SameLine()

            if psim.Button("- state"):
                sample = sample - np.prod(samples_grid[2:])

        if samples_grid[2] > 1:
            if psim.Button("+ surface"):
                sample = sample + np.prod(samples_grid[3:])

            psim.SameLine()

            if psim.Button("- surface"):
                sample = sample - np.prod(samples_grid[3:])

        if samples_grid[3] > 1:
            if psim.Button("+ rotation"):
                sample = sample + np.prod(samples_grid[4:])

            psim.SameLine()

            if psim.Button("- rotation"):
                sample = sample - np.prod(samples_grid[4:])

        if psim.Button("+"):
            sample = sample + 1

        psim.SameLine()

        if psim.Button("-"):
            sample = sample - 1

        sample = np.clip(sample, 0, num_samples-1)

        psim.SameLine()

        _, play = psim.Checkbox("play samples", play)

        if play:
            if sample < num_samples-1:
                sample = sample + 1
            else:
                sample = 0

        if show_near_nodes:
            _, dataset.near_nodes = psim.Checkbox("show near nodes", dataset.near_nodes)

        _, show_model = psim.Checkbox("show model", show_model)

        if show_model:

            psim.TextUnformatted(f"Model eval time(ms) : {1000*eval_time}")

            _, sleep = psim.Checkbox("sleep", sleep)

            if num_model_checkpoints != 0:
                _, model_num = psim.SliderInt("model checkpoint", model_num, 0, num_model_checkpoints-1)

            if model.descriptor_net.grid_size_descriptor == 0:

                _, model.descriptor_net.frame.num_samples = \
                    psim.SliderInt("frame samples", model.descriptor_net.frame.num_samples, 1, 1000)

                changed = psim.BeginCombo("frame mode", model.descriptor_net.frame.mode_frame.name)
                if changed:
                    for val in models.ModeFrame:
                        _, selected = psim.Selectable(val.name, model.descriptor_net.frame.mode_frame==val)
                        if selected:
                            model.descriptor_net.frame.mode_frame = val
                    psim.EndCombo()

                _, model.descriptor_net.frame.biased = \
                    psim.Checkbox("frame biased", model.descriptor_net.frame.biased)
            else:

                    _, model.post_interpolate = psim.Checkbox("post interpolate", model.post_interpolate)

            _, custom_collider = psim.Checkbox("custom collider", custom_collider)

            if custom_collider:

                if len(dataset_shapes) > 1:
                    _, shape = psim.SliderInt("shape", shape, 0, len(dataset_shapes)-1)

                _, pos[0] = psim.SliderFloat("translation x", pos[0], -2.0, 2.0)
                _, pos[1] = psim.SliderFloat("translation y", pos[1], -2.0, 2.0)
                _, pos[2] = psim.SliderFloat("translation z", pos[2], -2.0, 2.0)

                _, rot_axis[0] = psim.SliderFloat("rotation x", rot_axis[0], -np.pi, np.pi)
                _, rot_axis[1] = psim.SliderFloat("rotation y", rot_axis[1], -np.pi, np.pi)
                _, rot_axis[2] = psim.SliderFloat("rotation z", rot_axis[2], -np.pi, np.pi)

        if model_num_old != model_num and num_model_checkpoints != 0:
            state_dict_save = torch.load(model_file / "checkpoints" / f"checkpoint-{model_num}.pth", map_location=device)
            model.collider_net.load_state_dict(state_dict_save)

            model_num_old = model_num

        if sample_old != sample or not sleep:

            q, z, s, x, x_target = dataset[sample]

            if sample_old != sample:
                sample_old = sample

            ###### Show collider

            if show_model and custom_collider:

                rot = R.from_rotvec(rot_axis)
                z[0:3,:] = torch.from_numpy(rot.as_matrix().T)
                z[3,:] = torch.from_numpy(pos)
                s = shape

            if s_old != s:
                V_collider, F_collider = dataset_shapes[s]
                F_collider = F_collider.cpu().detach().numpy()

                W_collider = torch.cat((V_collider, torch.ones((V_collider.shape[0],1))), dim=1)
                x_collider = torch.matmul(W_collider, z.cpu()).cpu().detach().numpy()

                collider_mesh = ps.register_surface_mesh("collider", x_collider, F_collider, smooth_shade=False)

                s_old = s
            else:
                x_collider = torch.matmul(W_collider, z.cpu()).cpu().detach().numpy()

                collider_mesh.update_vertex_positions(x_collider)

            ###### Show object data

            x_reduced = x.cpu().detach().numpy()
            x_full = x_target.cpu().detach().numpy()

            if dataset.near_nodes and show_near_nodes:
                reduced_nodes = ps.register_point_cloud("reduced nodes", x_reduced)
                full_nodes = ps.register_point_cloud("full nodes", x_full)
                full_nodes.add_vector_quantity("deformation", x_reduced - x_full, 
                                               enabled=True, vectortype="ambient", color=full_color)

                if full_mesh != None:
                    ps.remove_surface_mesh("reduced")
                    ps.remove_surface_mesh("full")
                    reduced_mesh = None
                    full_mesh = None
            else:
                if full_mesh == None:
                    reduced_mesh = ps.register_surface_mesh("reduced", x_reduced, F_object)
                    full_mesh = ps.register_surface_mesh("full", x_full, F_object)
                else:
                    reduced_mesh.update_vertex_positions(x_reduced)
                    full_mesh.update_vertex_positions(x_full)

                if full_nodes != None:
                    ps.remove_point_cloud("reduced nodes")
                    ps.remove_point_cloud("full nodes")
                    reduced_nodes = None
                    full_nodes = None

            ###### Show object model

            if show_model:
            
                q = q.to(device)
                z = z.to(device)
                x = x.to(device)

                t_ini = time.time()
                x_model = model(q, z, s, x)[0].cpu().detach().numpy()
                t_fin = time.time()

                eval_time = t_fin - t_ini

                if dataset.near_nodes and show_near_nodes:
                    model_nodes = ps.register_point_cloud("model nodes", x_model)
                    model_nodes.add_vector_quantity("deformation", x_reduced - x_model, 
                                                    enabled=True, vectortype="ambient", color=model_color)

                    if model_mesh != None:
                        ps.remove_surface_mesh("model")
                        model_mesh = None
                else:
                    if model_mesh == None:
                        model_mesh = ps.register_surface_mesh("model", x_model, F_object)
                    else:
                        model_mesh.update_vertex_positions(x_model)

                    if model_nodes != None:
                        ps.remove_point_cloud("model nodes")
                        model_nodes = None

            else:

                eval_time = 0.0

                if model_mesh != None:
                    ps.remove_surface_mesh("model")
                    model_mesh = None
                if model_nodes != None:
                    ps.remove_point_cloud("model nodes")
                    model_nodes = None  

    ps.set_user_callback(main_loop)
    ps.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers()


    parser_record = subparser.add_parser('record', help='recording mode')

    parser_record.add_argument('--contact_problem', type = str, required=True,
                               help='Name of the contact problem to be evaluated')

    parser_record.add_argument('--model_file', type = pathlib.Path, default=None,
                               help='Path to trained model weights file (.pth). \
                                     If nothing is provided, model weights are randomly initialized')
    
    parser_record.add_argument('--precompute_descriptor', action='store_true',
                               help='Optional grid precomputation of the collider descritor')

    parser_record.add_argument('--output_path', type = pathlib.Path, required=True,
                               help='Path to the desired output directory. Different vertex position sequence \
                                     files (.pc2) are recorded for the collider, reduced deformation, full ground truth \
                                     and model results')


    parser_visualize = subparser.add_parser('visualize', help='visualization mode')

    parser_visualize.add_argument('--contact_problem', type = str, required=True,
                                  help='Name of the contact problem to be evaluated')

    parser_visualize.add_argument('--model_file', '--log_path', type = pathlib.Path, default=None,
                                  help='Path to trained model weights file (.pth). \
                                        Optionally, path to a log directory generated during training. \
                                        If nothing is provided, model weights are randomly initialized')
    
    parser_visualize.add_argument('--precompute_descriptor', action='store_true',
                                  help='Optional grid precomputation of the collider descritor')
    
    parser_visualize.add_argument('--show_near_nodes', action='store_true',
                                  help='Optional flag to precompute and visualize the nodes near and inside the collider. \
                                        Used for debuging the subset of nodes in the supervised training')

    args = parser.parse_args()

    evaluate(**vars(args))