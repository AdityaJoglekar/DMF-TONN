import tensorflow as tf
import numpy as np

class PINN():
    def __init__(self, problem, disp_model):
        self.problem = problem
        self.disp_model = disp_model
    @tf.function
    def pinn_loss(self,xPhys_m, coord):
        u = self.disp_model(coord)
        u = self.problem.analytical_fixed_BC(u,coord)

        u1 = u[:,0:1]
        u0 = u[:,1:2]
        u2 = u[:,2:3]
        uy_xyz = tf.gradients(u1,coord)[0]
        ux_xyz = tf.gradients(u0,coord)[0]
        uz_xyz = tf.gradients(u2,coord)[0]
        
        eps11 = ux_xyz[:,1]
        eps12 = 0.5 * ux_xyz[:,0] + 0.5 * uy_xyz[:,1]
        eps13 = 0.5 * ux_xyz[:,2] + 0.5 * uz_xyz[:,1]
        eps22 = uy_xyz[:,0]
        eps23 = 0.5 * uy_xyz[:,2] + 0.5 * uz_xyz[:,0]
        eps33 = uz_xyz[:,2]

        youngs_modulus = 1000.0
        poissons_ratio = 0.3
        lame_mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
        lame_lambda = youngs_modulus * poissons_ratio / \
            ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
        trace_strain = eps11 + eps22 + eps33
        squared_diagonal = eps11 * eps11 + eps33 * eps33 + eps22 * eps22
        energy = 0.5 * lame_lambda * trace_strain * trace_strain + lame_mu * \
            (squared_diagonal + 2.0 * eps12 * eps12 +
            2.0 * eps13 * eps13 + 2.0 * eps23 * eps23)
        energy = tf.reshape(energy,[-1,1])*(tf.math.pow(xPhys_m,3.0))
        energy_c = energy
        energy_ans =self.problem.V*tf.reduce_mean(energy)

        force_l = tf.reduce_mean(tf.matmul(self.disp_model(self.problem.dlX_force),self.problem.F_vector))
        loss = (energy_ans - force_l)

        return loss, energy_c