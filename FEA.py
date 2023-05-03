import tensorflow as tf
import time
import numpy as np
from scipy.sparse import coo_matrix
import cupy as cp
import cupyx as cpx
from cupyx.scipy.sparse import linalg as linalg_g

class StructuralFE:

    def __init__(self, nelx,nely,nelz):
        self.nely, self.nelx, self.nelz = nely,nelx,nelz
        self.nele = nelx*nely*nelz
        self.penal = 3.0


    def getDMatrix(self):
        E=1000
        nu=0.3
        A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8], [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]],dtype = float)
        k = A.T@np.array([[1],[nu]])/144.0
        K1 = np.array([k[0],k[1],k[1],k[2],k[4],k[4],k[1],k[0],k[1],k[3],k[5],k[6],k[1],k[1],k[0],k[3],k[6],k[5],k[2],k[3],k[3],k[0],k[7],k[7],k[4],k[5],k[6],k[7],k[0],k[1],k[4],k[6],k[5],k[7],k[1],k[0]])
        K2 = np.array([k[8],k[7],k[11],k[5],k[3],k[6],k[7],k[8],k[11],k[4],k[2],k[4],k[9],k[9],k[12],k[6],k[3],k[5],k[5],k[4],k[10],k[8],k[1],k[9],k[3],k[2],k[4],k[1],k[8],k[11],k[10],k[3],k[5],k[11],k[9],k[12]])
        K3 = np.array([k[5],k[6],k[3],k[8],k[11],k[7],k[6],k[5],k[3],k[9],k[12],k[9],k[4],k[4],k[2],k[7],k[11],k[8],k[8],k[9],k[1],k[5],k[10],k[4],k[11],k[12],k[9],k[10],k[5],k[3],k[1],k[11],k[8],k[3],k[4],k[2]])
        K4 = np.array([k[13],k[10],k[10],k[12],k[9],k[9],k[10],k[13],k[10],k[11],k[8],k[7],k[10],k[10],k[13],k[11],k[7],k[8],k[12],k[11],k[11],k[13],k[6],k[6],k[9],k[8],k[7],k[6],k[13],k[10],k[9],k[7],k[8],k[6],k[10],k[13]])
        K5 = np.array([k[0],k[1],k[7],k[2],k[4],k[3],k[1],k[0],k[7],k[3],k[5],k[10],k[7],k[7],k[0],k[4],k[10],k[5],k[2],k[3],k[4],k[0],k[7],k[1],k[4],k[5],k[10],k[7],k[0],k[7],k[3],k[10],k[5],k[1],k[7],k[0]])
        K6 = np.array([k[13],k[10],k[6],k[12],k[9],k[11],k[10],k[13],k[6],k[11],k[8],k[1],k[6],k[6],k[13],k[9],k[1],k[8],k[12],k[11],k[9],k[13],k[6],k[10],k[9],k[8],k[1],k[6],k[13],k[6],k[11],k[1],k[8],k[10],k[6],k[13]])
        K1 = np.reshape(K1,[6,6])
        K2 = np.reshape(K2,[6,6])
        K3 = np.reshape(K3,[6,6])
        K4 = np.reshape(K4,[6,6])
        K5 = np.reshape(K5,[6,6])
        K6 = np.reshape(K6,[6,6])
        KE = np.concatenate([np.concatenate([K1, K2, K3, K4],1),np.concatenate([K2.T, K5, K6, K3.T],1),np.concatenate([K3.T, K6, K5.T, K2.T],1),np.concatenate([K4, K3, K2, K1],1)],0)/((nu+1)*(1-2*nu))
        return KE
    #-----------------------#
    def initializeSolver(self, force_voxel,iif,jf,kf,Emin = 1e-3, Emax = 1000.0):
        self.Emin = Emin;
        self.Emax = Emax;

        idy1,idx1,idz1 = np.where(force_voxel)[0][0], np.where(force_voxel)[1][0],np.where(force_voxel)[2][0]
        idy1 = self.nely - 1 - idy1

        #User defined loads
        il = np.array([idx1+1,0,0],dtype = float)
        jl = np.array([idy1+1,0,0],dtype = float)
        kl = np.array([idz1+1,0,0],dtype = float)
        il_F = np.array([0,0,0],dtype = float)
        jl_F = np.array([0.1,0,0],dtype = float)
        kl_F = np.array([0,0,0],dtype = float)
        loadnid = kl*(self.nelx+1)*(self.nely+1) + il*(self.nely+1)+(self.nely+1-jl)
        loaddofx = 3*loadnid-2
        loaddofy = 3*loadnid-1
        loaddofz = 3*loadnid

        loaddofs = np.concatenate([loaddofx,loaddofy,loaddofz],0)
        loaddofs = np.concatenate([np.expand_dims(loaddofs-1,1),np.zeros([9,1])],1)
        loaddofv = np.concatenate([il_F,jl_F,kl_F],0)
        ndof = 3*(self.nelx+1)*(self.nely+1)*(self.nelz+1)

        F = coo_matrix((loaddofv, (loaddofs.astype(int)[:,0],loaddofs.astype(int)[:,1])), shape = [ndof,1])

        #User defined support fixed dofs
        # iif, jf,kf = np.meshgrid(np.linspace(0.0,0.0,1),np.linspace(0,self.nely,self.nely+1),np.linspace(0.0,self.nelz,self.nelz+1))
        fixednid = kf*(self.nelx+1)*(self.nely+1)+iif*(self.nely+1)+(self.nely+1-jf)
        fixeddof = np.concatenate([3*fixednid,3*fixednid-1,3*fixednid-2],1)
        # U = np.zeros([ndof,1])



        freedofs = np.setdiff1d(np.linspace(1.0,ndof,ndof), fixeddof.reshape([fixeddof.size]))
        # KE = lk_H8_np(nu)
        lele = 0.025
        self.KE=self.getDMatrix()*lele
        nodegrd = np.transpose(np.reshape(np.linspace(1.0,(self.nely+1)*(self.nelx+1),(self.nely+1)*(self.nelx+1)),[self.nelx+1,self.nely+1]))
        nodeids = np.reshape(np.transpose(nodegrd[0:self.nely,0:self.nelx]),[self.nely*self.nelx,1])
        nodeidz = np.linspace(0.0,(self.nelz-1)*(self.nely+1)*(self.nelx+1),(self.nelz))
        nodeids = nodeids*np.ones([1,nodeidz.shape[0]])+np.ones([self.nely*self.nelx,1])*nodeidz
        edofVec = 3*np.reshape(np.transpose(nodeids),[np.size(nodeids),1])+1
        self.edofMat = edofVec*np.ones([1,24]) + np.ones([self.nele,1])*np.array([0,1,2,3*self.nely+3,3*self.nely+4,3*self.nely+5,3*self.nely,3*self.nely+1,3*self.nely+2,-3,-2,-1,
                            3*(self.nely+1)*(self.nelx+1),3*(self.nely+1)*(self.nelx+1)+1,3*(self.nely+1)*(self.nelx+1)+2,3*(self.nely+1)*(self.nelx+1)+(3*self.nely+3),
                            3*(self.nely+1)*(self.nelx+1)+(3*self.nely+4),3*(self.nely+1)*(self.nelx+1)+(3*self.nely+5),3*(self.nely+1)*(self.nelx+1)+(3*self.nely),
                            3*(self.nely+1)*(self.nelx+1)+(3*self.nely+1),3*(self.nely+1)*(self.nelx+1)+(3*self.nely+2),3*(self.nely+1)*(self.nelx+1)-3,3*(self.nely+1)*(self.nelx+1)-2,
                            3*(self.nely+1)*(self.nelx+1)-1],dtype=float)
        iK = np.reshape(np.repeat(self.edofMat,24*np.ones([self.nele],dtype=int),axis = 0),[24*24*self.nele])
        jK = np.reshape(np.repeat(self.edofMat,24*np.ones([24],dtype=int),axis = 1),[24*24*self.nele])
        self.freedofs = np.reshape(freedofs-1,[np.size(freedofs)]).astype(int)

        #send necessary data into gpu
        self.iK_g = cp.array(iK)
        self.jK_g = cp.array(jK)
        F_g = cpx.scipy.sparse.coo_matrix((cp.array(loaddofv), (cp.array(loaddofs.astype(int)[:,0]),cp.array(loaddofs.astype(int)[:,1]))), shape = (ndof,1))
        self.F_f_g = F_g.tocsc()[self.freedofs]
        self.KE_g = cp.array(self.KE)
        self.freedofs_g = cp.array(self.freedofs)
        self.U_prev_g = cp.zeros(self.freedofs.shape)




    #-----------------------#
    @tf.custom_gradient
    def compliance(self, density):
        xPhys = cp.array(density.numpy())
        # tf.print(xPhys.shape[0])
        start = time.time()
        sK = cp.reshape(self.KE_g,[-1,1])*(self.Emin + cp.power(cp.reshape(cp.transpose(xPhys),[1,-1]),self.penal)*(self.Emax-self.Emin))
        sK = cp.reshape(cp.transpose(sK),[24*24*self.nele,1])
        K = cpx.scipy.sparse.coo_matrix((sK.reshape([-1]), ((self.jK_g-1).astype(int), (self.iK_g-1).astype(int))))
        K_f = K.tocsc()[self.freedofs] 
        K_f = K_f.transpose()[self.freedofs]
        f_size = self.freedofs.size
        K_fdiag = cp.asarray(K_f[cp.linspace(0,f_size-1,f_size,dtype=int),cp.linspace(0,f_size-1,f_size,dtype=int)])
        M =  cpx.scipy.sparse.spdiags(data = 1/K_fdiag,diags = 0, m = K_fdiag.size,n = K_fdiag.size)
        # print ('GPU Assembly took {} sec'.format(time.time()-start))
        start = time.time()
        if self.nele>5000:
            U_f,_ = linalg_g.cg(K_f,self.F_f_g.toarray(),x0=self.U_prev_g,M=M,maxiter = 8000)
        else:
            U_f = linalg_g.spsolve(K_f,self.F_f_g.toarray())
        self.U_prev_g = U_f
        # print ('GPU Linear solver took {} sec'.format(time.time()-start))
        U = cpx.scipy.sparse.coo_matrix((U_f, (self.freedofs_g,cp.zeros(self.freedofs.size,dtype = int)))).toarray()
        U_e = U[cp.reshape(self.edofMat-1,[-1]).astype(int)].reshape([self.edofMat.shape[0],self.edofMat.shape[1]])
        ce = cp.sum(U_e@self.KE_g*U_e,axis=1)
        c = self.Emax*tf.reduce_sum(tf.pow(xPhys.get(),self.penal)*tf.convert_to_tensor(ce.get().reshape([self.nely,self.nelx,self.nelz],order='F'),dtype=tf.float32))
        def grad(dy):
            dc = -dy*self.penal*tf.pow(xPhys.get(),self.penal-1.0)*tf.convert_to_tensor(ce.get().reshape([self.nely,self.nelx,self.nelz],order='F'),dtype=tf.float32)
            return tf.reshape(dc,[1,-1])
        return tf.convert_to_tensor(c,dtype=tf.float32),grad