import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
from functools import partial
from itertools import product
from tensorflow import keras
from config import cla
from data_gen import * 
from models import *
from utils import *
from pde_utils import *


print('\n=========================== STARTING TRAINER ============================\n')

PARAMS = cla()
np.random.seed(PARAMS.seed_no)

save_dir = f"exps/{PARAMS.save_dir}"   

print(f'--- Creating save directory: {save_dir}')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print('    *** directory aleady exists!!')


print(f'\n--- Saving parameters to file')
param_file = save_dir + '/parameters.txt'
with open(param_file,"w") as fid:
    for pname in vars(PARAMS):
        fid.write(f"{pname} = {vars(PARAMS)[pname]}\n")    

print(f'\n--- Loading data')
dataset = CreateDataset(save_dir=save_dir)

Nx1   = dataset.Nx1
Nx2   = dataset.Nx2
Nx1_all   = dataset.Nx1_all
Nx2_all   = dataset.Nx2_all
Z_dim = PARAMS.z_dim
x1L,x1R,x2L,x2R = dataset.x1[0],dataset.x1[-1],dataset.x2[0],dataset.x2[-1]
X_range = [x1L,x1R,x2L,x2R]



#======== Models ==============
# Everything is hard-coded for 2D PDEs

print(f'\n--- Creating network architectures')

if PARAMS.g_type == 'Standard':
    g_model = generator_std(X_dim=2,Z_dim=Z_dim,output_dim=1,width=PARAMS.g_width,depth=PARAMS.g_depth)
elif PARAMS.g_type == 'FF':
    B = tf.random.normal(mean=0,stddev=PARAMS.FF_sigma,shape=[PARAMS.FF_len,2]) 
    g_model = generator_FF(X_dim=2,Z_dim=Z_dim,output_dim=1,width=PARAMS.g_width,depth=PARAMS.g_depth,B=B)   

g_optim = tf.keras.optimizers.Adam(0.0001,beta_1=0.5, beta_2=0.9)
g_model.summary()

if PARAMS.d_type == 'MLP':
    d_model = critic_MLP(input_dim=Nx1*Nx2,width=PARAMS.d_width,depth=PARAMS.d_depth)
elif PARAMS.d_type == 'CNN':
    d_model = critic_CNN(input_H=Nx1,input_W=Nx2)
d_optim = tf.keras.optimizers.Adam(0.0001,beta_1=0.5, beta_2=0.9)
d_model.summary()


print(f'\n--- Creating nodal grids')
# ======== Meshing and Data loading =========== 
[X2,X1]   = np.meshgrid(dataset.x2,dataset.x1)
X         = tf.constant(np.hstack((X1.reshape(-1,1),X2.reshape(-1,1))),dtype=tf.float32) 
[X2_,X1_] = np.meshgrid(dataset.x2_data,dataset.x1_data)
X_        = tf.constant(np.hstack((X1_.reshape(-1,1),X2_.reshape(-1,1))),dtype=tf.float32)

glv = partial(get_lat_var,z_dim=Z_dim)
di  = partial(d_input,d_type=PARAMS.d_type)

MC  = 100
Z_  = glv(batch_size=MC,H=dataset.Nx1_all,W=dataset.Nx2_all)

ones   = tf.ones([PARAMS.batch_size, 1])
zeros  = tf.zeros([PARAMS.batch_size, 1])

train_dataset = dataset.dataset.shuffle(PARAMS.Nu).batch(PARAMS.batch_size, drop_remainder=True)


print(f'\n--- Training GAN')
# ============ Training ==================
n_iters  = 1
d_loss_log  = []
g_loss_log  = []
wd_loss_log = []
pdecon_log  = [[] for i in range(len(PARAMS.pdecon_coef))]



for i in range(PARAMS.n_epoch):
    
    for true in train_dataset:

        Z = glv(batch_size=PARAMS.batch_size,H=Nx1,W=Nx2)

        with tf.GradientTape() as tape:

            fake     = g_model([tf.tile(X,[PARAMS.batch_size,1]),Z],training=True)
            fake     = tf.reshape(fake,(-1,Nx1,Nx2))
            fake_val = d_model(di(fake))
            true_val = d_model(di(true))

            gp = gradient_penalty(real =di(true), 
                                  fake =di(fake), 
                                  model=d_model, 
                                  k    =PARAMS.gp_coef, 
                                  p    =2)
    
            # WGAN-div
            fake_loss = tf.reduce_mean(fake_val) 
            true_loss = tf.reduce_mean(true_val)

            wd_loss = true_loss - fake_loss

            d_loss = -wd_loss + gp 
        

        d_grad = tape.gradient(d_loss,d_model.trainable_variables)
        d_optim.apply_gradients(zip(d_grad,d_model.trainable_variables))

        print(f"iter:{n_iters} ---> d_loss:{d_loss.numpy():.4e}, gp_term:{gp.numpy():.4e}, wd:{wd_loss.numpy():.4e}")

        del tape

        if n_iters % PARAMS.n_critic == 0 :

            with tf.GradientTape() as tape:
                fake     = g_model([tf.tile(X,[PARAMS.batch_size,1]),Z],training=True)
                fake     = tf.reshape(fake,(-1,Nx1,Nx2))
                fake_val = d_model(di(fake))    
        
                # WGAN-div
                g_base_loss = -tf.reduce_mean(fake_val)
                g_loss      = g_base_loss

                if len(PARAMS.pdecon_coef) > 0:
                    constriants = pde_constraints(pde_type=PARAMS.pde_type,
                                                  model=g_model,
                                                  z_dim=Z_dim,
                                                  pde_params=PARAMS.pde_params,
                                                  X_range=X_range,
                                                  N=5000) # Number of random inputs to evaluate PDE constraints

                    assert len(constriants) == len(PARAMS.pdecon_coef)
                

                    for c in range(len(PARAMS.pdecon_coef)):
                        g_loss += PARAMS.pdecon_coef[c]*constriants[c]

            gen_grad = tape.gradient(g_loss,g_model.trainable_variables)
            g_optim.apply_gradients(zip(gen_grad,g_model.trainable_variables))

            
            print(f"               ---> g_loss:{g_loss.numpy():.4e}")
            for c in range(len(constriants)): 
                print(f"               ---> pde_constraint{c+1}:{constriants[c].numpy():.4e}")
                # pdecon_log[c].append(constriants[c].numpy())


            del tape
            
        n_iters += 1
   
    d_loss_log.append(d_loss.numpy())
    wd_loss_log.append(wd_loss.numpy())
    g_loss_log.append(g_loss.numpy())
    for c in range(len(constriants)): 
        pdecon_log[c].append(constriants[c].numpy())
    if (i==0) or ((i+1) % PARAMS.savefig_freq == 0):

        pred    = g_model([tf.tile(X_,[MC,1]),Z_],training=False)
        pred    = tf.reshape(pred,[MC,Nx1_all,Nx2_all]).numpy()
        mean    = np.mean(pred,axis=0)     
        std     = np.std(pred,axis=0)   

        plot_samples(pred,dataset.x1_data,dataset.x2_data,save_dir,f'samples_{i+1}')
        plot_stats(mean,std,dataset.u_mean,dataset.u_std,dataset.x1_data,dataset.x2_data,save_dir,f'stats_{i+1}')

        g_model.save_weights(f'{save_dir}/checkpoints/g_checkpoint_{i+1}')




# # ======== Plotting and Stuff ===========

save_loss(d_loss_log,'d_loss',save_dir,PARAMS.n_epoch)
save_loss(wd_loss_log,'wd_loss',save_dir,PARAMS.n_epoch)
save_loss(g_loss_log,'g_loss',save_dir,PARAMS.n_epoch)
for c in range(len(PARAMS.pdecon_coef)):
    save_loss(pdecon_log[c],f'pde_loss_{c}',save_dir,PARAMS.n_epoch)

print('\n=========================== FINISHED ============================\n')


