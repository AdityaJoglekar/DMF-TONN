import tensorflow as tf
import numpy as np


class Disp_Net():
    def __init__(self):
        low_band = 0.0
        high_band = 35
        c_y, c_x, c_z=np.meshgrid(np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
                                                    np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
                                                    np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),indexing='ij')
        dlInit = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 0)
        self.kernel1 = tf.Variable(dlInit,trainable=True,dtype=tf.float32)
        self.weights1 = tf.Variable(tf.zeros([dlInit.shape[1],3]),trainable=True)

    def __call__(self,coord):
        layer1 = tf.sin(tf.matmul(coord,  self.kernel1 ) + tf.ones([1,self.kernel1.shape[1]]))
        u = tf.matmul(layer1, self.weights1)
        return u   
              
    def get_weights(self):
        return [self.weights1]
    
class TO_Net():
    def __init__(self):
        low_band = 0.0
        high_band = 35
        c_y, c_x, c_z=np.meshgrid(np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
                                                    np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),
                                                    np.linspace([-high_band,low_band],[-low_band,high_band],8).reshape([-1]),indexing='ij')
        dlInit = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 0)
        self.kernel1 = tf.Variable(dlInit,trainable=True,dtype=tf.float32)
        self.weights1 = tf.Variable(tf.zeros([dlInit.shape[1],1]),trainable=True)

    def __call__(self,coord):
        layer1 = tf.sin(tf.matmul(coord, 1.0* self.kernel1 ) + tf.ones([1,self.kernel1.shape[1]]))
        rho = tf.nn.sigmoid(tf.matmul(layer1, self.weights1))
        return rho
    
    def get_weights(self):
        return [self.weights1]