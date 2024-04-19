import torch
from torch import nn
import numpy as np

import src.utils as utils

from enum import Enum

class ModeFrame(Enum):
    RANDOM = 0
    RANDOM_TANGENT = 1
    CURVATURE = 2
    CURVATURE_MINMAX = 3

class Frame(nn.Module):

    def __init__(self, sdf, radius, num_samples, mode_frame, biased_frame):
        super(Frame, self).__init__()

        self.sdf = sdf
        self.radius = radius
        self.num_samples = num_samples

        self.mean = nn.Parameter(torch.tensor(0.0), requires_grad = False)
        self.std = nn.Parameter(torch.tensor(1.0), requires_grad = False)
        self.distribution = torch.distributions.Normal(self.mean, self.std)

        self.mode_frame = mode_frame

        self.biased = biased_frame

        self.seed_num = 0
        self.eps = 1e-7

    def forward(self, x, s):

        if self.biased:
            torch.manual_seed(self.seed_num)

        num_shapes = x.shape[0]
        num_points = x.shape[1]
        dim = x.shape[2]

        def random_frames(radius, num_samples, normalized=False):

            if dim == 2:

                vec = radius * self.distribution.sample((num_samples, 2))

                if normalized:
                    vec_norm = torch.sqrt(torch.sum(vec * vec, dim=1)) + self.eps
                    vec = vec / vec_norm[:,None]

                frames = torch.stack((vec, vec[:,[1,0]]), dim=1)
                frames[:,1,0] = -frames[:,1,0]

            else:

                vec = self.distribution.sample((num_samples, 3))
                bivec = self.distribution.sample((num_samples, 3))

                vec_norm = torch.sqrt(torch.sum(vec * vec, dim=1)) + self.eps

                cross = torch.cross(vec, bivec, dim=1)
                cross_norm = torch.sqrt(torch.sum(cross * cross, dim=1)) + self.eps

                cross = cross * (vec_norm / cross_norm)[:,None]

                cross2 = torch.cross(vec, cross, dim=1) / vec_norm[:,None]

                frames = radius*torch.stack((vec, cross, cross2), dim=1)

                if normalized:
                    frames = frames / (vec_norm[:,None,None] * radius)

            return frames

        if self.mode_frame.value == 0:

            ###########
            # Random frame (Infinite cases)

            if self.biased:
                frames = random_frames(self.radius, num_shapes, normalized = True)
                frames = frames.reshape((num_shapes, 1, dim, dim))
                frames = frames.repeat(1, num_points, 1, 1)
            else: 
                frames = random_frames(self.radius, num_shapes*num_points, normalized = True)
                frames = frames.reshape(num_shapes, num_points, dim, dim)

        else:

            if self.biased:
                rand_frames = random_frames(self.radius, num_shapes*self.num_samples)
                rand_frames = rand_frames.reshape((num_shapes, 1, self.num_samples, dim, dim))
                rand_frames = rand_frames.repeat(1,num_points, 1, 1, 1)
            else:
                rand_frames = random_frames(self.radius, num_shapes*num_points*self.num_samples)
                rand_frames = rand_frames.reshape(num_shapes, num_points, self.num_samples, dim, dim)

            norm2 = torch.sum(rand_frames * rand_frames, dim=4) + self.eps

            ih_plus = self.sdf((x[:,:,None,None,:]+rand_frames).reshape(num_shapes, -1, dim), s).reshape(num_shapes, num_points, self.num_samples, dim)
            ih_minus = self.sdf((x[:,:,None,None,:]-rand_frames).reshape(num_shapes, -1, dim), s).reshape(num_shapes, num_points, self.num_samples, dim)

            diffs = (ih_plus - ih_minus) / norm2

            diffs_frames = diffs[:,:,:,:,None] * rand_frames

            grad = torch.sum(diffs_frames, dim=(2,3))

            norm_grad = torch.sqrt(torch.sum(grad * grad, dim=2)) + self.eps
            grad = grad / norm_grad[:,:,None]

            if dim == 2:
                frames = torch.stack((grad[:,:,[1,0]],grad), dim=2)
                frames[:,:,0,1] = -frames[:,:,0,1]

            else:

                ###########
                # Random tangent frame (Infinite cases)

                if self.biased:
                    bivec = self.distribution.sample((grad.shape[0], 1, 3))
                    bivec = bivec.repeat(1,num_points, 1)
                    bivec[:,:,:] = 0.0
                    bivec[:,:,0] = 1.0
                else:
                    bivec = self.distribution.sample((grad.shape[0], grad.shape[1], 3))
                        
                cross = torch.cross(grad, bivec, dim=2)

                cross_norm = torch.sqrt(torch.sum(cross * cross, dim=2)) + self.eps

                cross = cross / cross_norm[:,:,None]

                cross2 = torch.cross(grad, cross, dim=2)

                frames = torch.stack((grad,cross,cross2), dim=2)

                if self.mode_frame.value > 1:

                    ###########
                    # Curvature aligned frame (4 cases min)

                    i_center = self.sdf(x, s)

                    diffs = (ih_plus - 2*i_center[:,:,None,None] + ih_minus) / (norm2 * norm2)
                    outer = torch.einsum('ijklm,ijkln->ijklmn', rand_frames, rand_frames)
                    diffs_outer = diffs[:,:,:,:,None,None] * outer
                    hessian = torch.sum(diffs_outer, dim=(2,3))

                    frames_tangent = torch.stack((cross,cross2), dim=2)

                    hessian_proj = torch.matmul(frames_tangent, torch.matmul(hessian, torch.swapaxes(frames_tangent, -1, -2)))

                    eig_vals, eig_vecs = torch.linalg.eig(hessian_proj)
                    eig_vals = eig_vals.real
                    eig_vecs = eig_vecs.real

                    if self.mode_frame.value > 2:

                        ###########
                        # Curvature oriented frame (2 cases min)

                        #Order (rotate) eigenvectors based on absolute magnitude of eigenvalues
                        #The first eigenvector is going to be the direction of "maximum" curvature

                        _, indices = torch.sort(torch.abs(eig_vals), dim=-1, descending=True)

                        eig_vecs = torch.gather(eig_vecs, 3, indices[:,:,None,:].expand(-1, -1, 2, -1))
                        eig_vecs[:,:,:,0] = (1 - 2*indices[:,:,0])[:,:,None] * eig_vecs[:,:,:,0]

                    vecs = torch.swapaxes(eig_vecs, -1, -2)

                    frames[:,:,1:,:] = torch.matmul(vecs, frames_tangent)
 
        return frames

class Patch(nn.Module):

    def __init__(self, sdf, frame, radius, pattern):
        super(Patch, self).__init__()

        self.sdf = sdf
        self.frame = frame

        self.num_features = pattern.shape[0]
        self.pattern = nn.Parameter(radius*pattern.float(), requires_grad = False)

    def forward(self, x, s):

        num_shapes = x.shape[0]
        num_points = x.shape[1]
        dim = x.shape[2]

        rot_transposed = self.frame(x, s)
        feat_points = torch.matmul(self.pattern[None,None,:,:], rot_transposed) + x[:,:,None,:]

        feat_sdf = self.sdf(feat_points.reshape(num_shapes,-1,dim), s)
            
        patches = feat_sdf.reshape((num_shapes, num_points, self.num_features))

        return patches

class FieldDescriptorNet(nn.Module):

    def __init__(self, dataset_shapes, patch_pattern, grid_size_sdf=50, grid_size_descriptor=50, 
                 mode_frame=ModeFrame.RANDOM, biased_frame=False, samples_per_frame=50,
                 frame_radius=0.2, patch_radius=0.2, max_distance=0.2):
        super(FieldDescriptorNet, self).__init__()

        self.grid_size_sdf = grid_size_sdf
        self.grid_size_descriptor = grid_size_descriptor

        center, bounding_radius = utils.getBoundingSphere(dataset_shapes)

        self.center = nn.Parameter(center, requires_grad = False)
        self.bounding_radius = nn.Parameter(bounding_radius, requires_grad = False)

        sampling_radius = max(frame_radius, patch_radius)

        radius_sdf = bounding_radius + max_distance + sampling_radius 
        radius_feat = bounding_radius + max_distance

        sdf = utils.SDF(dataset_shapes)
        
        if grid_size_sdf == 0:
            self.sdf = sdf
        else:
            print("computing sdf interpolator...")
            self.sdf = utils.GridInterpolator(sdf, center, radius_sdf, grid_size_sdf, extend=True)

        frame = Frame(self.sdf, frame_radius, samples_per_frame, mode_frame, biased_frame)

        if grid_size_descriptor == 0:
            self.frame = frame
        else:
            print("computing frame interpolator...")
            self.frame = utils.GridInterpolator(frame, center, radius_feat, grid_size_descriptor, extend=False, normalize=True)

        patch = Patch(self.sdf, self.frame, patch_radius, patch_pattern)

        if grid_size_descriptor == 0:
            self.patch = patch
        else:
            print("computing patch interpolator...")
            self.patch = utils.GridInterpolator(patch, center, radius_feat, grid_size_descriptor, extend=True)

    def forward(self, x, s):
        
        frames = self.frame(x, s)
        patches = self.patch(x, s)

        return patches, frames

    def gridValues(self, x, s):

        frames = self.frame.gridValues(x, s)
        patches = self.patch.gridValues(x, s)

        return patches, frames

    def gridWeighting(self, x, s, values):
        
        interp_values = self.patch.gridWeighting(x, s, values)

        return interp_values

class CorrectionNet(nn.Module):
    
    def __init__(self, dim, dim_params, dim_paramsShape):
        super(CorrectionNet, self).__init__()
        
        self.dim_paramsShape = dim_paramsShape

        self.mean_params = nn.Parameter(torch.zeros(dim_params, dim), requires_grad = False)
        self.std_params = nn.Parameter(torch.ones(dim_params, dim), requires_grad = False)
        self.mean_paramsShape = nn.Parameter(torch.zeros(dim_paramsShape), requires_grad = False)
        self.std_paramsShape = nn.Parameter(torch.ones(dim_paramsShape), requires_grad = False)

        self.num_activations = [150,150,150,150,50,10]
        self.num_activations.append(dim)
        
        self.linear1_params = nn.Linear(dim_params*dim, self.num_activations[0])
        self.linear1_paramsShape = nn.Linear(dim_paramsShape, self.num_activations[0], bias=False)

        lineartemp = nn.Linear(dim_params*dim + dim_paramsShape, self.num_activations[0])
        self.linear1_params.weight.data = lineartemp.weight.data[:,:dim_params*dim]
        self.linear1_params.bias.data = lineartemp.bias.data
        self.linear1_paramsShape.weight.data = lineartemp.weight.data[:,dim_params*dim:]

        self.act = nn.ModuleList([])
        self.linear = nn.ModuleList([])

        for i in range(len(self.num_activations) - 1):
            self.act.append( nn.ELU() )
            self.linear.append( nn.Linear(self.num_activations[i], self.num_activations[i+1]) )

        self.mean_out = nn.Parameter(torch.zeros(dim), requires_grad = False)
        self.std_out = nn.Parameter(torch.ones(dim), requires_grad = False)

    def clearMean(self):
        
        self.mean_params[:,:] = 0.0
        self.mean_paramsShape[:] = 0.0
        self.mean_out[:] = 0.0
        
    def accumMean(self, params, shape_params, out):
        
        self.mean_params.data += torch.mean(params, axis=(0,1))
        self.mean_paramsShape.data += torch.mean(shape_params, axis=(0,1))
        self.mean_out.data += torch.mean(out, axis=(0,1))
        
    def divideMean(self, size):

        self.mean_params.data /= size
        self.mean_paramsShape.data /= size
        self.mean_out.data /= size

        print("Params Mean: ", self.mean_params.data) 
        print("ParamsShape Mean: ", self.mean_paramsShape.data)
        print("Out Mean: ",self.mean_out.data)

    def clearStd(self):   
        
        self.std_params[:,:] = 0.0
        self.std_paramsShape[:] = 0.0
        self.std_out[:] = 0.0

    def accumStd(self, params, shape_params, out):
        
        delparams = params - self.mean_params.data[None,None,:,:]
        self.std_params.data += torch.mean(delparams * delparams, axis=(0,1))

        delshape = shape_params - self.mean_paramsShape.data[None,None,:]
        self.std_paramsShape.data += torch.mean(delshape * delshape, axis=(0,1))

        delout = out - self.mean_out.data[None,None,:]
        self.std_out.data += torch.mean(delout * delout, axis=(0,1))

    def divideStd(self, size):
        
        self.std_params.data /= size
        self.std_paramsShape.data /= size
        self.std_out.data /= size

        self.std_params.data = torch.sqrt(self.std_params.data)
        self.std_paramsShape.data = torch.sqrt(self.std_paramsShape.data)
        self.std_out.data = torch.sqrt(self.std_out.data)

        print("Params Std: ", self.std_params.data)
        print("ParamsShape Std: ", self.std_paramsShape.data)
        print("Out Std: ",self.std_out.data)

    def forward(self, params, shape_params):
        
        nparams = params - self.mean_params
        nparams = nparams / self.std_params

        if self.dim_paramsShape == 0:
            r = self.linear1_params(torch.flatten(nparams, start_dim=-2))
        else:
            nshape_params = shape_params - self.mean_paramsShape
            nshape_params = nshape_params / self.std_paramsShape

            r = self.linear1_paramsShape(nshape_params)
            r = r + self.linear1_params(torch.flatten(nparams, start_dim=-2))

        for i in range(len(self.linear)):

            r = self.act[i](r)
            r = self.linear[i](r)

        r = r * self.std_out
        r = r + self.mean_out

        return r

class DefModel(nn.Module):
    
    def __init__(self, path_object, dataset_shapes, patch_pattern,
                 grid_size_sdf=50, grid_size_descriptor=50, mode_frame=ModeFrame.RANDOM,
                 biased_frame=False, samples_per_frame=50,
                 frame_radius=0.2, patch_radius=0.2, max_distance=0.2, mask_factor=0.3):
        super(DefModel, self).__init__()

        numbers_object = np.load(path_object / "info" / "numbers.npy")
        dim = numbers_object[1]
        num_rigids = numbers_object[2]
        num_points = numbers_object[3]

        self.dim = dim
        self.num_rigids = num_rigids
        self.num_points = num_points

        self.near_eval = True
        self.near_eval_threshold = 0.1
        self.post_interpolate = True

        self.descriptor_net = FieldDescriptorNet(dataset_shapes, patch_pattern, grid_size_sdf, grid_size_descriptor, 
                                                 mode_frame, biased_frame, samples_per_frame,
                                                 frame_radius, patch_radius, max_distance)

        self.collider_net = CorrectionNet(dim, num_rigids*(dim+1) + num_points, patch_pattern.shape[0])

        self.eyeb = nn.Parameter(torch.eye(dim).reshape((1, 1, dim, dim)), requires_grad = False)

        self.max_distance = max_distance
        self.mask_factor = mask_factor

        self.tanh = nn.Tanh()
        
    def transformInput(self, qs, zs, ss, xs):
        
        # get z rotation and translation
        z_rotT = zs[:,0:self.dim,:]
        z_rot = torch.swapaxes(z_rotT, -1, -2)
        z_trans = zs[:,self.dim::,:]

        # transform x to collider frame
        x_relative = torch.matmul((xs - z_trans), z_rot)

        # get patch local features
        patch_shape, patch_rotT_local = self.descriptor_net(x_relative, ss)

        # update patch local rotation with collider rotation
        patch_rotT = torch.matmul(patch_rotT_local, z_rotT[:,None,:,:])
        patch_rot = torch.swapaxes(patch_rotT, -1, -2)

        # transform q to patch local frame
        q_patch_relative = torch.matmul(qs, patch_rot)

        return q_patch_relative, patch_shape, patch_rotT

    def getLocalTransform(self, q, z, s, x, x_target):
        
        # prepare batch dimension
        num_verts = x.shape[-2]

        qs = q.reshape((-1, num_verts, (self.dim+1)*self.num_rigids + self.num_points, self.dim))
        zs = z.reshape((-1, self.dim+1, self.dim))
        ss = s
        xs = x.reshape((-1, num_verts, self.dim))
        x_targets = x_target.reshape((-1, num_verts, self.dim))

        # transform input
        q_patch_relative, patch_shape, patch_rotT = self.transformInput(qs, zs, ss, xs)
        
        # transform to local correction
        patch_rot = torch.swapaxes(patch_rotT, -1, -2)
        x_corr_local = torch.matmul((x_targets - xs).unsqueeze(-2), patch_rot).squeeze(-2)
 
        return q_patch_relative, patch_shape, x_corr_local

    def mask(self, zs, ss, xs, scaling=1.0):

        # get z rotation and translation
        z_rotT = zs[:,0:self.dim,:]
        z_rot = torch.swapaxes(z_rotT, -1, -2)
        z_trans = zs[:,self.dim::,:]

        # transform x to collider frame
        x_relative = torch.matmul((xs - z_trans), z_rot) / scaling

        # evaluate sdf
        d = scaling * self.descriptor_net.sdf(x_relative, ss)
        
        # compute attenuation mask
        scale = self.mask_factor*self.max_distance
        ds = (d + scale - self.max_distance) / scale
        atten = 0.5 - 0.5*self.tanh(ds)

        return d, atten

    def forward(self, q, z, s, x):
        
        # prepare batch dimension
        num_verts = x.shape[-2]

        qs = q.reshape((-1, num_verts, (self.dim+1)*self.num_rigids + self.num_points, self.dim))
        zs = z.reshape((-1, self.dim+1, self.dim))
        ss = s
        xs = x.reshape((-1, num_verts, self.dim))

        # Init output
        
        # compute near nodes mask

        if not self.training:

            _, atten = self.mask(zs, ss, xs)

        if self.near_eval and not self.training:

            atten_mask = atten > self.near_eval_threshold

            qs = qs[atten_mask].reshape((1, -1, (self.dim+1)*self.num_rigids + self.num_points, self.dim))

            xs_corr = xs.clone()
            xs = xs[atten_mask].reshape((1, -1, self.dim))

            atten = atten[atten_mask].reshape((1, -1))
        
        # Transform input

        # get z rotation and translation
        z_rotT = zs[:,0:self.dim,:]
        z_rot = torch.swapaxes(z_rotT, -1, -2)
        z_trans = zs[:,self.dim::,:]

        # transform x to collider frame
        x_relative = torch.matmul((xs - z_trans), z_rot)

        #
        if self.post_interpolate and self.descriptor_net.grid_size_descriptor != 0:

            # get patch local features
            patch_shape, patch_rotT_local = self.descriptor_net.gridValues(x_relative, ss)

            # update patch local rotation with collider rotation
            patch_rotT = torch.matmul(patch_rotT_local, z_rotT[:,None,:,:])
            patch_rot = torch.swapaxes(patch_rotT, -1, -2)

            # transform q to patch local frame

            num_grid_neighbours = 2**self.dim

            q_patch_relative = torch.matmul(qs.repeat(1,num_grid_neighbours,1,1), patch_rot)

            # compute local correction
            corr = self.collider_net(q_patch_relative, patch_shape)

            # transform to global correction
            corr = torch.matmul(corr.unsqueeze(-2), patch_rotT).squeeze(-2)

            # interpolate corrections
            corr = self.descriptor_net.gridWeighting(x_relative, ss, corr)

        else:

            # get patch local features
            patch_shape, patch_rotT_local = self.descriptor_net(x_relative, ss)

            # update patch local rotation with collider rotation
            patch_rotT = torch.matmul(patch_rotT_local, z_rotT[:,None,:,:])
            patch_rot = torch.swapaxes(patch_rotT, -1, -2)

            # transform q to patch local frame
            q_patch_relative = torch.matmul(qs, patch_rot)

            # compute local correction
            corr = self.collider_net(q_patch_relative, patch_shape)

            # transform to global correction
            corr = torch.matmul(corr.unsqueeze(-2), patch_rotT).squeeze(-2)


        if not self.training:
            corr = corr * atten[:,:,None]

        if self.near_eval and not self.training:
            xs_corr[atten_mask] += corr.squeeze(-3)
        else:   
            xs_corr = xs + corr

        return xs_corr