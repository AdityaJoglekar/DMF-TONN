import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from IPython import display
import numpy as np
from skimage import measure
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_disp(DMF_TONN,uPhys):
        plt.figure(figsize = (15,5))
        # y displacement field at z = 0
        plt.subplot(1,3,1)
        uPhysy = (tf.reshape(uPhys[:,0],[DMF_TONN.problem.nely,DMF_TONN.problem.nelx,DMF_TONN.problem.nelz]))
        plt.imshow(uPhysy[:,:,0])
        plt.colorbar()
        plt.title('u_y')

        # x displacement field at z = 0
        plt.subplot(1,3,2)
        uPhysx = (tf.reshape(uPhys[:,1],[DMF_TONN.problem.nely,DMF_TONN.problem.nelx,DMF_TONN.problem.nelz]))
        plt.imshow(uPhysx[:,:,0])
        plt.colorbar()
        plt.title('u_x')

        # z displacement field at z = 0
        plt.subplot(1,3,3)
        uPhysx = (tf.reshape(uPhys[:,2],[DMF_TONN.problem.nely,DMF_TONN.problem.nelx,DMF_TONN.problem.nelz]))
        plt.imshow(uPhysx[:,:,0])
        plt.colorbar()
        plt.title('u_z')
        plt.show()

def plot_params(DMF_TONN):
    plt.figure(figsize=(20,5))
    plt.subplot(1,4,1)
    plt.title('compliance loss vs Iteration')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    
    plt.plot(DMF_TONN.log_c)
    plt.subplot(1,4,2)
    plt.title('PINN loss vs Iteration')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(np.array(DMF_TONN.log_disp_loss))

    plt.subplot(1,4,3)
    plt.title('FE Compliance vs Iteration')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(np.array(DMF_TONN.log_fec))

    plt.subplot(1,4,4)
    plt.title('Volume Fraction vs Iteration')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(np.array(DMF_TONN.log_vf))
    plt.show()

def plot_xPhys(DMF_TONN,xPhys):
    xPhys = tf.reshape(xPhys,[DMF_TONN.problem.nely,DMF_TONN.problem.nelx,DMF_TONN.problem.nelz])
    plt.imshow(tf.reduce_mean(xPhys,axis=2),vmin=0, vmax=1,cmap = 'seismic')
    plt.show()

def plot3d(DMF_TONN,xPhys):

    cutoff = 0.4
    to_output = tf.reshape(xPhys,[DMF_TONN.problem.nely,DMF_TONN.problem.nelx,DMF_TONN.problem.nelz])
    fig = plt.figure(figsize=(15,15))
    voxelarray = to_output.numpy()
    voxelarray[np.where(voxelarray<cutoff)] = 0
    voxelarray[np.where(voxelarray>cutoff)] = 1
    voxelarray.astype(bool)
    voxelarray = np.flip(voxelarray,axis=0)
    ax = fig.add_subplot(1,1,1,projection='3d')
    norm = colors.Normalize(vmin=0.0, vmax=1.2, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.seismic)
    color = mapper.to_rgba(xPhys.numpy()).reshape([DMF_TONN.problem.nely,DMF_TONN.problem.nelx,DMF_TONN.problem.nelz,4])
    color = np.flip(color,axis=0)
    ls = LightSource(azdeg=45, altdeg=-45)
    ax.voxels(voxelarray,facecolors=color, edgecolor='k',lightsource = ls)
    ax.view_init(30, -60,vertical_axis='x')
    ax.set_box_aspect(aspect = (DMF_TONN.problem.nelx,DMF_TONN.problem.nelz,DMF_TONN.problem.nely))
    ax.set_axis_off()

def plot_iso(xPhys): #xPhys in shape [nely,nelx,nelz] and numpy array
    nely, nelx, nelz = xPhys.shape
    nelm = max(nely,nelx,nelz)
    padding = np.zeros([nely+2,nelx+2,nelz+2])
    padding[1:-1, 1:-1, 1:-1] = np.copy(xPhys)
    xPhys = padding
    verts, faces, normals, values = measure.marching_cubes(xPhys, 0.5) #set the density cutoff
    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ls = LightSource(azdeg=45, altdeg=-45)
    f_coord = np.take(verts, faces,axis = 0)
    f_norm = np.cross(f_coord[:,2] - f_coord[:,0], f_coord[:,1] - f_coord[:,0])
    cl = ls.shade_normals(f_norm)
    norm = colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.gray_r)
    rgb = mapper.to_rgba(cl).reshape([-1,4])
    mesh = Poly3DCollection(np.take(verts, faces,axis = 0)/nelx * (np.array([[nelm/nely, nelm/nelm, nelm/nelz]])))
    ax.add_collection3d(mesh)
    mesh.set_facecolors(rgb)
    ax.view_init(-150, -120,vertical_axis='x')
    ax.set_box_aspect(aspect = (nelx,nelz,nely))
    ax.set_axis_off()
    plt.show()

def save_result(DMF_TONN,directory):
    mode = "p"
    header = directory+mode+"_"+"{:.2f}".format(DMF_TONN.problem.volfrac)+"_"+"{:02d}".format(DMF_TONN.problem.xid)+"_"+"{:02d}".format(DMF_TONN.problem.yid)+"_"+"{:02d}".format(DMF_TONN.problem.zid)+"_"
    np.save(header+"log_xPhys.npy",np.array(DMF_TONN.log_xPhys))
    np.save(header+"log_c.npy",np.array(DMF_TONN.log_fec))
    np.save(header+"log_vf.npy",np.array(DMF_TONN.log_vf))
    np.save(header+"w.npy",np.array(DMF_TONN.to_model.weights1))
    np.save(header+"k.npy",np.array(DMF_TONN.to_model.kernel1))