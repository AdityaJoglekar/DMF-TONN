import tensorflow as tf
import numpy as np

tf.random.set_seed(1)
np.random.seed(1)

class Problems():
    def dlX_disp(self):
        domain_xcoord = np.random.uniform(-self.nelx/(2*(self.nelm)),self.nelx/(2*(self.nelm)),(self.batch_size - self.dlX_fixed.shape[0] - self.dlX_force.shape[0],1))
        domain_ycoord = np.random.uniform(-self.nely/(2*(self.nelm)),self.nely/(2*(self.nelm)),(self.batch_size - self.dlX_fixed.shape[0] - self.dlX_force.shape[0],1))
        domain_zcoord = np.random.uniform(-self.nelz/(2*(self.nelm)),self.nelz/(2*(self.nelm)),(self.batch_size - self.dlX_fixed.shape[0] - self.dlX_force.shape[0],1))
        domain_coord = np.concatenate((domain_ycoord,domain_xcoord,domain_zcoord),axis = 1)
        coord = np.concatenate((self.dlX_fixed, self.dlX_force),axis = 0)
        coord = np.concatenate((coord, domain_coord),axis = 0)
        coord = tf.convert_to_tensor(coord,dtype=tf.float32)
        return coord

class Cantilever_Beam_3D(Problems):
    def __init__(self,nelx, nely, nelz, xid, yid, zid, vf):
        
        self.xid = xid
        self.yid = yid
        self.zid = zid
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.nele = self.nelx*self.nely*self.nelz
        self.nelm = max(self.nelx,self.nely,self.nelz)
        self.volfrac = vf
        self.E0 = 1000
        self.nu = 0.3
        
        self.batch_size = 6000

        self.alpha_init = 1
        self.alpha_max = 100
        self.alpha_delta = 0.5

        self.penal = 3.0
        
        c_y, c_x, c_z=np.meshgrid(np.linspace(-(self.nely)/(2*self.nelm),(self.nely)/(2*self.nelm),self.nely),
                                                np.linspace(-(self.nelx)/(2*self.nelm),(self.nelx)/(2*self.nelm),self.nelx),
                                                np.linspace(-(self.nelz)/(2*self.nelm),(self.nelz)/(2*self.nelm),self.nelz),indexing='ij')
        self.dlX = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 1).reshape([-1,3])
        c_y, c_x, c_z=np.meshgrid(np.linspace(-(self.nely)/(2*self.nelm),(self.nely)/(2*self.nelm),2*self.nely),
                                                np.linspace(-(self.nelx)/(2*self.nelm),(self.nelx)/(2*self.nelm),2*self.nelx),
                                                np.linspace(-(self.nelz)/(2*self.nelm),(self.nelz)/(2*self.nelm),2*self.nelz),indexing='ij')
        self.dlXSS = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 1).reshape([-1,3])
        self.V = (np.max(self.dlX[:,0])-np.min(self.dlX[:,0]))*(np.max(self.dlX[:,1])-np.min(self.dlX[:,1]))*(np.max(self.dlX[:,2])-np.min(self.dlX[:,2]))
        
        #Problem boundary condition
        fixed_voxel = np.zeros((self.nely,self.nelx,self.nelz))
        fixed_voxel[:,0,:] = 1.0
        fixed_voxel = fixed_voxel.reshape([self.nele,1])

        dlX_fixed = self.dlX[np.where(fixed_voxel == 1.0)[0],:]

        F = 0.1
        self.F_vector = tf.constant([[F],[0.0],[0.0]],dtype = tf.float32)
        self.force_voxel = np.zeros((self.nely,self.nelx,self.nelz)) 
        self.force_voxel[yid,xid,zid] = 1
        force_voxel = self.force_voxel.reshape([self.nele,1])
        dlX_force = self.dlX[np.where(force_voxel == 1)[0],:]

        self.dlX = tf.convert_to_tensor(self.dlX,dtype=tf.float32)
        self.dlXSS = tf.convert_to_tensor(self.dlXSS,dtype=tf.float32)
        self.dlX_fixed = tf.convert_to_tensor(dlX_fixed,dtype=tf.float32)
        self.dlX_force = tf.convert_to_tensor(dlX_force,dtype=tf.float32)

        self.iif, self.jf,self.kf = np.meshgrid(np.linspace(0.0,0.0,1),np.linspace(0,self.nely,self.nely+1),np.linspace(0.0,self.nelz,self.nelz+1))

    def analytical_fixed_BC(self,u,coord):
        u = u*2*(1/(1+tf.exp(-20*(coord[:,1:2]+0.5))) - 0.5)
        return u