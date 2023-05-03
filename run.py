"""
Created on Wed May  3 11:49:54 2023

"""
from problems import Cantilever_Beam_3D
from neural_networks import Disp_Net,TO_Net
from dmf_tonn import DMF_TONN
from util import plot3d,plot_disp,plot_params,plot_xPhys,save_result,plot_iso
import numpy as np
import tensorflow as tf


if __name__ =='__main__':

    #User defined number of elements in each direction (for: 1) defining ratios of sides in DMF-TONN 2) Comparison with SIMP)
    nelx = 40
    nely = 20
    nelz = 8

    #User defined load location
    xid = 39
    yid = 19
    zid = 3

    #User defined target volume fraction
    vf = 0.3
    
    #Initialize
    problem= Cantilever_Beam_3D(nelx,nely,nelz,xid,yid,zid,vf)
    to_model= TO_Net()
    disp_model_h = Disp_Net()
    opt = DMF_TONN(problem, to_model, disp_model_h)

    #Run initial displacement network training
    opt.fit_disp_init()

    #Run optimization
    opt.fit_to(700)

    #Render Result
    xPhys_dlX = opt.to_model(opt.problem.dlXSS)
    plot_iso(np.reshape(xPhys_dlX,(2*opt.problem.nely,2*opt.problem.nelx,2*opt.problem.nelz)))