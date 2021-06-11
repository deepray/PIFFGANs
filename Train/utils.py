import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

def get_lat_var(batch_size,H,W,z_dim):
    z = tf.repeat(tf.random.normal(shape=(batch_size,z_dim)),repeats=H*W,axis=0)
    return z    

def d_input(input,d_type):
    if d_type == 'MLP':
        return tf.reshape(input.shape[0],input.shape[1]*input.shape[2])
    elif d_type == 'CNN':
        return tf.expand_dims(input,-1) 



def mesh(start, finish, steps):
    x = np.linspace(start, finish, steps)
    y = np.linspace(start, finish, steps)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def save_loss(loss,loss_name,savedir,n_epoch):
    

    np.savetxt(f"{savedir}/{loss_name}.txt",loss)

    with sns.axes_style("darkgrid"):
        fig, ax1 = plt.subplots()
        ax1.plot(np.arange(1,n_epoch+1),loss,'-o')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel(loss_name)
        ax1.set_xlim([1,n_epoch])

        plt.tight_layout()
        plt.savefig(f"{savedir}/{loss_name}.png",dpi=200)  

        ax1.set_ylim([np.min(loss[(n_epoch//2)-1::]),np.max(loss[(n_epoch//2)-1::])]) 
        ax1.set_xlim([n_epoch//2,n_epoch]) 
        plt.tight_layout()
        plt.savefig(f"{savedir}/{loss_name}_tail.png",dpi=200)  
        plt.close()  

  
    
