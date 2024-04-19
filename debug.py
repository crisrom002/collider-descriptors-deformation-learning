
import argparse
import pathlib

import numpy as np
import torch

import src.datasets as datasets
import src.models as models
import src.utils as utils

import polyscope as ps
import polyscope.imgui as psim

def debug(collider_path: pathlib.Path):
    """Interactive visualization of the collider signed distance field and descriptor with debugging purposes.
       Provided a path to the collider data, the marching cubes surface reconstruction and the descriptor 
       frames can be visualized at different offsets. Different frame computation modes can be compared.

    Keyword arguments:

    collider_path: Path to directory with the collider data.
    """

    seed = 0
    torch.manual_seed(seed)
    print("Torch Seed: ", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch Device: ", device)

    ######

    dataset_shapes = datasets.ShapeDataset(collider_path)
    center_shapes, radius_shapes = utils.getBoundingSphere(dataset_shapes)

    pattern = utils.getPattern(collider_path.parents[1])
 
    descriptor = models.FieldDescriptorNet(dataset_shapes,
                                           patch_pattern = pattern,
                                           grid_size_sdf = 50,
                                           grid_size_descriptor = 0,
                                           mode_frame = models.ModeFrame.RANDOM_TANGENT,
                                           biased_frame = False,
                                           samples_per_frame = 50,
                                           frame_radius = 0.1,
                                           patch_radius = 0.2,
                                           max_distance = 0.15)
    descriptor.to(device)

    shape_ids = np.arange(len(dataset_shapes))
    grid_size_sampling = 50
    d, _, bounding_box, spacing = utils.getGridSampling(descriptor.sdf, 
                                                        center = center_shapes, 
                                                        radius = radius_shapes, 
                                                        grid_size = grid_size_sampling, 
                                                        shapes = shape_ids, 
                                                        device = device)

    d_min = d.reshape((d.shape[0],-1)).min(dim=1)[0].cpu().detach().numpy()
    d_max = d.reshape((d.shape[0],-1)).max(dim=1)[0].cpu().detach().numpy()

    # Visualization
    #############################################################

    sleep = False

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    
    shape = 0
    offset_mesh = 0.0
    offset_frame = 0.0

    def main_loop():

        nonlocal sleep, shape, offset_mesh, offset_frame

        _, shape = psim.SliderInt("shape", shape, 0, shape_ids.shape[0]-1)

        ds = torch.unsqueeze(d[shape], 0)
        bounding_boxs = torch.unsqueeze(bounding_box[shape], 0)
        spacings = torch.unsqueeze(spacing[shape], 0)

        psim.TextUnformatted(f"SDF grid size : {descriptor.grid_size_sdf}^3")
        psim.TextUnformatted(f"Sampling grid size : {grid_size_sampling}^3")

        _, offset_mesh = psim.SliderFloat("offset surface", offset_mesh, 0.9*d_min[shape], 0.3*d_max[shape])

        verts, faces = utils.marching_cubes(ds, bounding_boxs, spacings, offset_mesh, device=device)

        _, offset_frame = psim.SliderFloat("offset frame", offset_frame, 0.9*d_min[shape], 0.3*d_max[shape])

        verts2, _ = utils.marching_cubes(ds, bounding_boxs, spacings, offset_frame, device=device)

        if descriptor.grid_size_descriptor == 0:

            _, descriptor.frame.num_samples = \
            psim.SliderInt("frame samples", descriptor.frame.num_samples, 1, 200)

            changed = psim.BeginCombo("frame mode", descriptor.frame.mode_frame.name)
            if changed:
                for val in models.ModeFrame:
                    _, selected = psim.Selectable(val.name, descriptor.frame.mode_frame==val)
                    if selected:
                        descriptor.frame.mode_frame = val
                psim.EndCombo()

            _, descriptor.frame.biased = psim.Checkbox("frame biased", descriptor.frame.biased)

        _, sleep = psim.Checkbox("sleep", sleep)

        if not sleep:
            rot_transpose2 = descriptor.frame(torch.unsqueeze(verts2[0], 0), shape_ids[shape])

            verts = verts[0].cpu().detach().numpy()
            faces = faces[0].cpu().detach().numpy()
            verts2 = verts2[0].cpu().detach().numpy()
            rot_transpose2 = rot_transpose2[0].cpu().detach().numpy()

            collider_mesh = ps.register_surface_mesh("collider", verts, faces, smooth_shade=True)
            collider_mesh.set_color((1.0,1.0,1.0))

            frames_mesh = ps.register_point_cloud("frames", verts2, radius=0.0)

            frames_mesh.add_vector_quantity("X", rot_transpose2[:,0,:], enabled=True, color=(1.0, 0.0, 0.0))
            frames_mesh.add_vector_quantity("Y", rot_transpose2[:,1,:], enabled=True, color=(0.0, 1.0, 0.0))
            frames_mesh.add_vector_quantity("Z", rot_transpose2[:,2,:], enabled=True, color=(0.0, 0.0, 1.0))
            
    ps.set_user_callback(main_loop)
    ps.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--collider_path', type=pathlib.Path, required=True, 
                        help='Path to directory with the collider data')
    
    args = parser.parse_args()
    
    debug(**vars(args))