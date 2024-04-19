import pathlib
import torch
import numpy as np
import igl

class ShapeDataset(torch.utils.data.Dataset):

    def __init__(self, path_collider):

        self.path_collider = path_collider

        shape_path = self.path_collider / "shapes"
        self.num_shapes = len(list(shape_path.iterdir()))
    
    def __len__(self):
        
        return self.num_shapes

    def __getitem__(self, index):

        # Does not support batching! 
        # Each shape can have different number of vertices / edges / faces

        shape_file = self.path_collider / "shapes" / f"{index}.obj"

        shapeV, shapeF = igl.read_triangle_mesh(str(shape_file.absolute()))
 
        num_nodes = shapeV.shape[0]

        # If there is no face info in the obj, we assume a 2D object
        if shapeF.shape[0] == 0:

            shapeV = torch.from_numpy(shapeV.astype("float32"))
            shapeV = shapeV[:,0:2]
            
            # The edges are automatically generated based on the order of the nodes
            r = torch.arange(num_nodes)

            shapeF = torch.stack((r,r+1)).T
            shapeF[num_nodes-1,1] = 0

        else:

            shapeV = torch.from_numpy(shapeV.astype("float32"))
            shapeF = torch.from_numpy(shapeF.astype("int64"))

        return shapeV, shapeF

class ContactDataset(torch.utils.data.Dataset):

    def __init__(self, device, model, path_object, path_data, data_slice,
                 near_nodes = False, batch_nodes = 0):

        self.path_object = path_object
        self.path_data = path_data
        self.data_slice = data_slice
        self.near_nodes = near_nodes
        self.batch_nodes = batch_nodes

        self.W = np.load(path_object / "info" / "W.npy")
        numbers_object = np.load(path_object / "info" / "numbers.npy")

        self.num_nodes = numbers_object[0]
        dim = numbers_object[1]
        num_rigids = numbers_object[2]
        num_points = numbers_object[3]

        ids_rigid_trans = torch.arange((dim+1)*num_rigids)[dim::dim+1]
        ids_point_trans = (dim+1)*num_rigids + torch.arange(num_points)
        self.ids_trans = torch.cat((ids_rigid_trans, ids_point_trans))

        self.correspondence_state = np.load(path_data / "correspondenceState.npy").reshape(-1)
        self.correspondence_shape = np.load(path_data / "correspondenceShape.npy").reshape(-1)

        self.num_samples_state = len(list((path_object / "state").iterdir()))
        self.num_samples_collider = self.correspondence_state.shape[0]

        shape_state = (self.num_samples_state, num_rigids*(dim+1) + num_points, dim)
        shape_collider = (self.num_samples_collider, dim + 1, dim)
        shape_state_def = (self.num_samples_state, self.num_nodes, dim)
        shape_collider_def = (self.num_samples_collider, self.num_nodes, dim)

        if not (path_object / "state.mmap").is_file():
            
            state = np.memmap(path_object / "state.mmap", dtype='float32', mode='w+', shape=shape_state)
            state_def = np.memmap(path_object / "stateDef.mmap", dtype='float32', mode='w+', shape=shape_state_def)
        
            for i in range(self.num_samples_state):
                print("memory mapping state frame... ", i, " / ", self.num_samples_state , end='\r')
                state[i] = np.load(path_object / "state" / f"{i}.npy")
                state_def[i] = np.load(path_object / "stateDef" / f"{i}.npy")
            
            print(end='\n') 
        
            del state
            del state_def

        if not (path_data / "collider.mmap").is_file():
            
            collider = np.memmap(path_data / "collider.mmap", dtype='float32', mode='w+', shape=shape_collider)
            collider_def = np.memmap(path_data / "colliderDef.mmap", dtype='float32', mode='w+', shape=shape_collider_def)

            for i in range(self.num_samples_collider):
                print("memory mapping collider frame... ", i, " / ", self.num_samples_collider , end='\r')
                collider[i] = np.load(path_data / "collider" / f"{i}.npy")
                collider_def[i] = np.load(path_data / "colliderDef" / f"{i}.npy")
            
            print(end='\n') 

            del collider
            del collider_def

        self.state_npy = np.memmap(path_object / "state.mmap",
            dtype="float32",
            mode='r',
            shape=shape_state
            )

        self.collider_npy = np.memmap(path_data / "collider.mmap",
            dtype="float32",
            mode='r',
            shape=shape_collider
             )

        self.state_def_npy = np.memmap(path_object / "stateDef.mmap",
            dtype="float32",
            mode='r',
            shape=shape_state_def
             )

        self.collider_def_npy = np.memmap(path_data / "colliderDef.mmap",
            dtype="float32",
            mode='r',
            shape=shape_collider_def
             )

        if self.near_nodes:
            
            self.re = np.zeros((len(data_slice),2), dtype=int)
            self.idsF = np.zeros((0,2), dtype=int)
            
            for i in range(len(data_slice)):
                
                print("computing near nodes... ", i, " / ", len(data_slice) , end='\r')

                index = data_slice[i]
                index_state = self.correspondence_state[index]
                
                collider_npy = np.array(self.collider_npy[index,:,:])[None,:,:]
                state_def_npy = np.array(self.state_def_npy[index_state,:,:])[None,:,:]

                zs = torch.from_numpy(collider_npy).to(device)
                ss = int(self.correspondence_shape[index])
                xs = torch.from_numpy(state_def_npy).to(device)
                
                d, atten = model.mask(zs, ss, xs)

                d = d[0].cpu().numpy()
                atten = atten[0].cpu().numpy()

                mask = (np.random.rand(atten.shape[0]) < atten)

                ids = np.arange(self.num_nodes)[mask]
                ids = np.stack((index * np.ones(ids.shape, dtype=int), ids), axis=1)

                self.re[i,0] = self.idsF.shape[0]
                self.re[i,1] = ids.shape[0]

                self.idsF = np.concatenate((self.idsF, ids))

            print(end='\n') 

            print("num near nodes: ", self.idsF.shape[0])

    def __len__(self):

        if self.near_nodes:
            if self.batch_nodes != 0:
                return self.idsF.shape[0]
            else:
                return len(self.data_slice)
            
        else:
            if self.batch_nodes != 0:
                return len(self.data_slice) * self.num_nodes
            else:
                return len(self.data_slice)

    def __getitem__(self, index):

        if self.near_nodes:
            if self.batch_nodes != 0:
                i = self.idsF[index,0]
                ids = np.array([self.idsF[index,1]])
                ids = ids.repeat(self.batch_nodes)
            else:
                # Does not support batching!
                i = self.data_slice[index]
                offset = self.re[index,0]
                size = self.re[index,1]
                ids = np.array(self.idsF[offset:offset+size,1])
        else:
            if self.batch_nodes != 0:
                i = self.data_slice[index // self.num_nodes]
                ids = np.array([index % self.num_nodes])
                ids = ids.repeat(self.batch_nodes)
            else:
                i = self.data_slice[index]
                ids = np.arange(self.num_nodes)
        
        i_state = self.correspondence_state[i]
        
        state_npy = np.array(self.state_npy[i_state,:,:])
        collider_npy = np.array(self.collider_npy[i,:,:])
        state_def_npy = np.array(self.state_def_npy[i_state,ids,:])
        collider_def_npy = np.array(self.collider_def_npy[i,ids,:])

        num_verts = ids.shape
        state_npy_mod = np.expand_dims(state_npy, axis=0).repeat(num_verts, axis=0)
        state_npy_mod[:,self.ids_trans,:] = state_npy[None,self.ids_trans,:] - state_def_npy[:,None,:]

        state_npy_mod = self.W[ids,:,None] * state_npy_mod

        state = torch.from_numpy(state_npy_mod)
        collider = torch.from_numpy(collider_npy)
        shape = int(self.correspondence_shape[i])
        state_def = torch.from_numpy(state_def_npy)
        collider_def = torch.from_numpy(collider_def_npy)
        
        return state, collider, shape, state_def, collider_def