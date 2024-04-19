import numpy
import torch
from torch import nn
import igl
from skimage import measure

def getBoundingSphere(dataset_shapes):

    center = []
    radius = []

    num_shapes = len(dataset_shapes)

    for i in range(num_shapes):
            
        V,_ = dataset_shapes[i]

        maxV = torch.amax(V, dim=0)
        minV = torch.amin(V, dim=0)

        center.append( (maxV + minV) / 2.0 )
        radius.append( torch.norm((maxV - minV)) / 2.0 )

    return torch.stack(center, 0), torch.stack(radius, 0)

def getBoundingBox(center, radius):

    bounding_box = torch.stack((center - radius[:,None], center + radius[:,None]), dim=-2)

    return bounding_box

def getGridSampling(func, center, radius, grid_size, shapes, index="ij", device="cpu"):

    dim = center.shape[1]

    # Small scaling to prevent numerical precision errors!
    bounding_box = getBoundingBox(center, 1.01*radius) 
    spacing = (bounding_box[:,1] - bounding_box[:,0]) / (grid_size-1)

    valuesL = []
    pL = []

    for i in shapes:
            
        p = []

        for d in range(dim):

            x = torch.linspace(bounding_box[i,0,d], bounding_box[i,1,d], grid_size).to(device)
            p.append(x)

        pp = torch.meshgrid(p, indexing=index)
        xy = torch.stack(pp, dim=dim)

        p = xy.reshape((1, grid_size**dim, dim))
        z = func(p,i)

        valuesL.append(z)
        pL.append(p)

    values = torch.cat(valuesL, 0)
    p = torch.cat(pL, 0)

    p = p.reshape((shapes.shape[0],) + (grid_size,) * dim + (dim,))

    dims_value = values.shape[2:]
    values = values.reshape((shapes.shape[0],) + (grid_size,) * dim + dims_value)
    
    values = values.to(device)
    p = p.to(device)
    bounding_box = bounding_box.to(device)
    spacing = spacing.to(device)

    return values, p, bounding_box[shapes], spacing[shapes]

def marching_cubes(d, bounding_box, spacing, level, device="cpu"):

    num_shapes = bounding_box.shape[0]

    vertsL = [] 
    facesL = [] 

    for shape in range(num_shapes):

        dS = d[shape].cpu().detach().numpy()
        bounding_boxS = bounding_box[shape].cpu().detach().numpy()
        spacingS = spacing[shape].cpu().detach().numpy()

        verts, faces, _, _ = measure.marching_cubes(dS, level, spacing=spacingS)
        verts += bounding_boxS[0,None,:]

        verts = torch.from_numpy(verts.copy()).to(device)
        faces = torch.from_numpy(faces.copy()).to(device)

        vertsL.append(verts)
        facesL.append(faces)

    return vertsL, facesL

def getPattern(path_data, half=False):

    pattern = numpy.load(path_data / "patterns" / "ball_unit_07_3d.npy")
    pattern = numpy.concatenate((numpy.zeros((1,3)),pattern))

    pattern = torch.from_numpy(pattern)
    
    # Rotate to make more simetric around X axis
    patternT = pattern.clone()
    pattern[:,0] = patternT[:,2]
    pattern[:,2] = -patternT[:,0]

    if half:
        pattern = pattern[pattern[:,0] < 0.001]

    return pattern

class GridInterpolator(nn.Module):

    def __init__(self, func, center, radius, grid_size, extend = False, normalize = False):
        super(GridInterpolator, self).__init__()

        num_shapes = center.shape[0]
        self.dim = center.shape[1]

        self.center = nn.Parameter(center, requires_grad = False)
        self.radius = nn.Parameter(radius, requires_grad = False)

        self.extend = extend
        self.normalize = normalize

        # sample all shapes
        shapes = numpy.arange(num_shapes) 
        values, _, bounding_box, spacing = getGridSampling(func, center, radius, grid_size, shapes)

        self.values = nn.Parameter(values, requires_grad = False)
        self.bounding_box = nn.Parameter(bounding_box, requires_grad = False)
        self.spacing = nn.Parameter(spacing, requires_grad = False)

        # get the dimensions of the interpolated quantity
        self.dims_value_size = tuple(values.shape[1+self.dim:])
        self.dims_value_len = len(values.shape[1+self.dim:])

    def linearInterp(self, x, s):

        id_scalar = (x - self.bounding_box[s,0,None,:]) / self.spacing[s,None,:]

        ids_min = torch.floor(id_scalar).long()
        w = (id_scalar - ids_min)
        ids_max = ids_min+1
        nw = 1 - w

        if self.dim == 2:
            
            a = nw[:,:,0] * nw[:,:,1]
            b = w[:,:,0]  * nw[:,:,1]
            c = nw[:,:,0] * w[:,:,1]
            d = w[:,:,0]  * w[:,:,1]

            shape = (x.shape[0],) + (-1,) + (1,) * self.dims_value_len
            repeat = (1,) + (1,) + self.dims_value_size

            ar = a.reshape(shape).repeat(repeat)
            br = b.reshape(shape).repeat(repeat)
            cr = c.reshape(shape).repeat(repeat)
            dr = d.reshape(shape).repeat(repeat)

            res = ar * self.values[s[:,None], ids_min[:,:,0], ids_min[:,:,1]] + \
                  br * self.values[s[:,None], ids_max[:,:,0], ids_min[:,:,1]] + \
                  cr * self.values[s[:,None], ids_min[:,:,0], ids_max[:,:,1]] + \
                  dr * self.values[s[:,None], ids_max[:,:,0], ids_max[:,:,1]]

        else:
            
            a = nw[:,:,0] * nw[:,:,1] * nw[:,:,2]
            b = w[:,:,0]  * nw[:,:,1] * nw[:,:,2]
            c = nw[:,:,0] * w[:,:,1]  * nw[:,:,2]
            d = w[:,:,0]  * w[:,:,1]  * nw[:,:,2]
            e = nw[:,:,0] * nw[:,:,1] * w[:,:,2]
            f = w[:,:,0]  * nw[:,:,1] * w[:,:,2]
            g = nw[:,:,0] * w[:,:,1]  * w[:,:,2]
            h = w[:,:,0]  * w[:,:,1]  * w[:,:,2]

            shape = (x.shape[0],) + (-1,) + (1,) * self.dims_value_len
            repeat = (1,) + (1,) + self.dims_value_size

            ar = a.reshape(shape).repeat(repeat)
            br = b.reshape(shape).repeat(repeat)
            cr = c.reshape(shape).repeat(repeat)
            dr = d.reshape(shape).repeat(repeat)
            er = e.reshape(shape).repeat(repeat)
            fr = f.reshape(shape).repeat(repeat)
            gr = g.reshape(shape).repeat(repeat)
            hr = h.reshape(shape).repeat(repeat)

            res = ar * self.values[s[:,None], ids_min[:,:,0], ids_min[:,:,1], ids_min[:,:,2]] + \
                  br * self.values[s[:,None], ids_max[:,:,0], ids_min[:,:,1], ids_min[:,:,2]] + \
                  cr * self.values[s[:,None], ids_min[:,:,0], ids_max[:,:,1], ids_min[:,:,2]] + \
                  dr * self.values[s[:,None], ids_max[:,:,0], ids_max[:,:,1], ids_min[:,:,2]] + \
                  er * self.values[s[:,None], ids_min[:,:,0], ids_min[:,:,1], ids_max[:,:,2]] + \
                  fr * self.values[s[:,None], ids_max[:,:,0], ids_min[:,:,1], ids_max[:,:,2]] + \
                  gr * self.values[s[:,None], ids_min[:,:,0], ids_max[:,:,1], ids_max[:,:,2]] + \
                  hr * self.values[s[:,None], ids_max[:,:,0], ids_max[:,:,1], ids_max[:,:,2]]

        return res

    def forward(self, x, s):
        
        if not isinstance(s, numpy.ndarray) and not isinstance(s, torch.Tensor):
            s = numpy.array([s])

        d = x - self.center[s,None,:]

        normd = torch.norm(d, dim=2)/self.radius[s,None]
        normd = torch.clamp(normd, min=1.0, max=None)

        dclamp = d / normd[:,:,None]
        xclamp = dclamp + self.center[s,None,:]

        d = self.linearInterp(xclamp, s)

        if self.extend:
            
            shape = (x.shape[0],) + (-1,) + (1,) * self.dims_value_len
            repeat = (1,) + (1,) + self.dims_value_size

            d = d + torch.norm(x - xclamp, dim=2).reshape(shape).repeat(repeat)

        if self.normalize:
            
            # Assumes normalization is required in last dimension!
            d = d / torch.norm(d, dim=3)[:,:,:,None]

        return d

    def linearInterpValues(self, x, s):

        id_scalar = (x - self.bounding_box[s,0,None,:]) / self.spacing[s,None,:]

        ids_min = torch.floor(id_scalar).long()
        ids_max = ids_min+1

        if self.dim == 2:
            
            res = torch.cat((self.values[s[:,None], ids_min[:,:,0], ids_min[:,:,1]],
                             self.values[s[:,None], ids_max[:,:,0], ids_min[:,:,1]],
                             self.values[s[:,None], ids_min[:,:,0], ids_max[:,:,1]],
                             self.values[s[:,None], ids_max[:,:,0], ids_max[:,:,1]]), dim=1)

        else:
            
            res = torch.cat((self.values[s[:,None], ids_min[:,:,0], ids_min[:,:,1], ids_min[:,:,2]], 
                             self.values[s[:,None], ids_max[:,:,0], ids_min[:,:,1], ids_min[:,:,2]], 
                             self.values[s[:,None], ids_min[:,:,0], ids_max[:,:,1], ids_min[:,:,2]],
                             self.values[s[:,None], ids_max[:,:,0], ids_max[:,:,1], ids_min[:,:,2]],
                             self.values[s[:,None], ids_min[:,:,0], ids_min[:,:,1], ids_max[:,:,2]],
                             self.values[s[:,None], ids_max[:,:,0], ids_min[:,:,1], ids_max[:,:,2]],
                             self.values[s[:,None], ids_min[:,:,0], ids_max[:,:,1], ids_max[:,:,2]],
                             self.values[s[:,None], ids_max[:,:,0], ids_max[:,:,1], ids_max[:,:,2]]), dim=1)

        return res

    def linearInterpWeighting(self, x, s, values):

        id_scalar = (x - self.bounding_box[s,0,None,:]) / self.spacing[s,None,:]

        ids_min = torch.floor(id_scalar).long()
        w = (id_scalar - ids_min)
        nw = 1 - w

        if self.dim == 2:
            
            a = nw[:,:,0] * nw[:,:,1]
            b = w[:,:,0]  * nw[:,:,1]
            c = nw[:,:,0] * w[:,:,1]
            d = w[:,:,0]  * w[:,:,1]

            shape = (x.shape[0],) + (-1,) + (1,) * self.dims_value_len

            numpoints = values.shape[1] // 4

            res = a.reshape(shape) * values[:,0*numpoints:1*numpoints,:] + \
                  b.reshape(shape) * values[:,1*numpoints:2*numpoints,:] + \
                  c.reshape(shape) * values[:,2*numpoints:3*numpoints,:] + \
                  d.reshape(shape) * values[:,3*numpoints:4*numpoints,:]

        else:
            
            a = nw[:,:,0] * nw[:,:,1] * nw[:,:,2]
            b = w[:,:,0]  * nw[:,:,1] * nw[:,:,2]
            c = nw[:,:,0] * w[:,:,1]  * nw[:,:,2]
            d = w[:,:,0]  * w[:,:,1]  * nw[:,:,2]
            e = nw[:,:,0] * nw[:,:,1] * w[:,:,2]
            f = w[:,:,0]  * nw[:,:,1] * w[:,:,2]
            g = nw[:,:,0] * w[:,:,1]  * w[:,:,2]
            h = w[:,:,0]  * w[:,:,1]  * w[:,:,2]

            shape = (x.shape[0],) + (-1,) + (1,) * self.dims_value_len

            numpoints = values.shape[1] // 8

            res = a.reshape(shape) * values[:,0*numpoints:1*numpoints,:] + \
                  b.reshape(shape) * values[:,1*numpoints:2*numpoints,:] + \
                  c.reshape(shape) * values[:,2*numpoints:3*numpoints,:] + \
                  d.reshape(shape) * values[:,3*numpoints:4*numpoints,:] + \
                  e.reshape(shape) * values[:,4*numpoints:5*numpoints,:] + \
                  f.reshape(shape) * values[:,5*numpoints:6*numpoints,:] + \
                  g.reshape(shape) * values[:,6*numpoints:7*numpoints,:] + \
                  h.reshape(shape) * values[:,7*numpoints:8*numpoints,:]

        return res

    def gridValues(self, x, s):
        
        if not isinstance(s, numpy.ndarray) and not isinstance(s, torch.Tensor):
            s = numpy.array([s])

        d = x - self.center[s,None,:]

        normd = torch.norm(d, dim=2)/self.radius[s,None]
        normd = torch.clamp(normd, min=1.0, max=None)

        dclamp = d / normd[:,:,None]
        xclamp = dclamp + self.center[s,None,:]

        d = self.linearInterpValues(xclamp, s)

        return d

    def gridWeighting(self, x, s, values):

        if not isinstance(s, numpy.ndarray) and not isinstance(s, torch.Tensor):
            s = numpy.array([s])

        d = x - self.center[s,None,:]

        normd = torch.norm(d, dim=2)/self.radius[s,None]
        normd = torch.clamp(normd, min=1.0, max=None)

        dclamp = d / normd[:,:,None]
        xclamp = dclamp + self.center[s,None,:]

        d = self.linearInterpWeighting(xclamp, s, values)

        return d

class SDF(nn.Module):

    def __init__(self, dataset_shapes):
        super(SDF, self).__init__()

        self.V = nn.ParameterList([])
        self.F = nn.ParameterList([])

        for i in range(len(dataset_shapes)):

            V,F = dataset_shapes[i]

            self.V.append(nn.Parameter(V, requires_grad = False))
            self.F.append(nn.Parameter(F, requires_grad = False))

    def forward(self, x, s):

        num_shapes = x.shape[0]
        dim = x.shape[2]

        if not isinstance(s, numpy.ndarray) and not isinstance(s, torch.Tensor):
            s = numpy.array([s])

        sd = []

        for i in range(num_shapes):
            
            ss = s[i].item()
            V = self.V[ss]
            F = self.F[ss]

            # 2D surface
            if dim == 2:

                e = V[F[:,1]] - V[F[:,0]]

                d_norm2 = torch.sum(e * e, dim=1)
                en = e / d_norm2[:,None]
                r = x[i,:,None,:] - V[F[:,0]][None,:,:]

                t = torch.sum(r * en[None,:,:], dim=2)
                t = torch.clip(t, 0.0, 1.0)
                p = r - e[None,:,:] * t[:,:,None]

                d = torch.sum(p * p, dim=2)
                dmin = torch.amin(d, dim=1)

                b0 = x[i,:,None,1] >= V[F[:,0]][None,:,1]
                b1 = x[i,:,None,1] < V[F[:,1]][None,:,1]

                n = e[:, [1, 0]]
                n[:,1] *= -1.0
                b2 = torch.sum(n[None,:,:] * p, dim=2) < 0.0

                b = torch.stack((b0,b1,b2),dim=2)
                b_all = torch.all(b,dim=2)
                b_all_invert = torch.all(~b,dim=2)

                rr = torch.sum(b_all.long(), dim=1) - torch.sum(b_all_invert.long(), dim=1)
                rr = (rr%2).float()

                sd.append( (1.0-2.0*rr)*torch.sqrt(dmin) )

            #3D surface
            else:
                
                x_np = x[i].detach().cpu().numpy()
                V_np = V.detach().cpu().numpy()
                F_np = F.detach().cpu().numpy()

                d = igl.signed_distance(x_np, V_np, F_np)[0]
                
                device = next(self.parameters()).device
                d = torch.from_numpy(d).to(device)

                sd.append( d )  

        return torch.stack(sd, 0)