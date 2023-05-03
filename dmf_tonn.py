import tensorflow as tf
from IPython import display
import numpy as np
# from FEA import StructuralFE
from pinn import PINN


class DMF_TONN():

    def __init__(self, problem, to_model, disp_model):
        self.problem = problem
        self.disp_model = disp_model
        self.to_model = to_model
        self.log_vf = []
        self.log_disp_loss = []
        self.log_c = []
        self.log_pinn_init_loss = []
        self.log_fec = []
        self.log_xPhys = []
        self.pinn = PINN(self.problem, self.disp_model)
        self.total_epoch = 0

        self.disp_optimizer = tf.keras.optimizers.Adam(learning_rate=0.000005)
        self.to_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        # self.FE = StructuralFE(problem.nelx,problem.nely,problem.nelz)
        # self.FE.initializeSolver(problem.force_voxel,problem.iif,problem.jf,problem.kf,10e-6, 1000.0)

    def train_step_disp(self, xPhys_m,coord):
        with tf.GradientTape() as model_tape:
            loss,_ = self.pinn.pinn_loss(xPhys_m,coord)
            self.log_pinn_init_loss.append(loss)
        
        grad = model_tape.gradient(loss,self.disp_model.get_weights())
        self.disp_optimizer.apply_gradients(zip(grad, self.disp_model.get_weights()))

    def fit_disp_init(self):
        epochs = 1000
        for epoch in range(epochs):
            
            if epoch%100 ==1:
                display.clear_output(wait=True)
                # plt.figure(figsize = (20,10))
                # plt.title('PINN loss vs Iteration',fontsize = 20)
                # plt.xlabel('Iteration',fontsize = 20)
                # plt.ylabel('Loss',fontsize = 20)
                # plt.tick_params(axis='both', labelsize=20)
                # plt.plot(np.array(self.log_pinn_init_loss))
                # plt.show()
                # uPhys =self.disp_model(self.problem.dlX)
                # plot_disp(self,uPhys)
                
            coord = self.problem.dlX_disp()
            xPhys = tf.ones([coord.shape[0],1])*0.5
            with tf.GradientTape() as model_tape:
                loss,energy_c = self.pinn.pinn_loss(xPhys,coord)
                self.log_pinn_init_loss.append(loss)
            grad = model_tape.gradient(loss,self.disp_model.get_weights())
            self.disp_optimizer.apply_gradients(zip(grad, self.disp_model.get_weights()))
        xPhys = tf.ones([self.problem.dlX.shape[0],1])*0.5
        loss,energy_c = self.pinn.pinn_loss(xPhys,self.problem.dlX)
        self.c_0 = tf.reduce_mean(energy_c)

    @tf.custom_gradient
    def compute_de_drho(self, xPhys_m, energy_c,coord):
        def gradient_function(denergy):
            with tf.GradientTape(persistent = True) as grad_tape:
                grad_tape.watch(xPhys_m)
                loss,energy_c = self.pinn.pinn_loss(xPhys_m,coord)
            gradients = denergy*grad_tape.gradient(energy_c, xPhys_m)
            return -gradients, tf.zeros_like(energy_c), tf.zeros_like(coord)
        return energy_c, gradient_function
    
    def to_loss(self, coord):
 
        self.total_epoch = self.total_epoch+1

        xPhys_m = self.to_model(coord)

        alpha = min(self.problem.alpha_init + self.problem.alpha_delta * self.total_epoch , self.problem.alpha_max)

        _,energy_c = self.pinn.pinn_loss(xPhys_m,coord)
        c = tf.reduce_mean(self.compute_de_drho(xPhys_m,energy_c,coord))
        xPhys_dlX = self.to_model(self.problem.dlX)
        vf = tf.math.reduce_mean(xPhys_dlX)
        loss = 1.0*c/self.c_0+alpha*(vf/self.problem.volfrac-1.0)**2 
        # fe_c = self.FE.compliance(tf.reshape(xPhys_dlX,[self.problem.nely,self.problem.nelx,self.problem.nelz]))
        # fe_c = 0

        tf.print('Epoch:',self.total_epoch)
        # tf.print('Compliance:',c)
        # tf.print('Compliance from FE:',fe_c)
        # tf.print('VF:',vf)
        tf.print('Total Loss:',loss)
        
        # self.log_fec.append(fe_c)
        self.log_c.append(c)
        self.log_vf.append(vf)
        self.log_xPhys.append(xPhys_dlX)
        return loss    

    def fit_disp(self, epochs=200):

        for i in range(epochs):
            coord = self.problem.dlX_disp()
            with tf.GradientTape() as model_tape:
                xPhys_m = self.to_model(coord)
                loss,_ = self.pinn.pinn_loss(xPhys_m,coord)
                self.log_disp_loss.append(loss)
            grad = model_tape.gradient(loss,self.disp_model.get_weights())
            self.disp_optimizer.apply_gradients(zip(grad, self.disp_model.get_weights()))

    def fit_to(self, epochs):
        for epoch in range(epochs):
            self.fit_disp(20)
            if epoch%10 == 1:
                display.clear_output(wait=True)
                # xPhys_m = self.to_model(self.problem.dlX)
                # uPhys = self.disp_model(self.problem.dlX) 
                # plot_xPhys(self,xPhys_m)
                # plot3d(self,xPhys_m)
                # plot_disp(self,uPhys)
                # plot_params(self)

            coord = self.problem.dlX_disp()
            with tf.GradientTape() as model_tape:
                loss = self.to_loss(coord)

            grad = model_tape.gradient(loss,self.to_model.get_weights())
            self.to_optimizer.apply_gradients(zip(grad, self.to_model.get_weights()))