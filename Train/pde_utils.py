import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import tensorflow as tf
from tensorflow import keras
from models import *

np.random.seed(1008)

@tf.function
def pde_constraints(pde_type,model,z_dim,pde_params,X_range,N):

    if pde_type == 'None':
        constraints = []
    elif pde_type == 'LinAdv':
        constraints = linadv_constraints(model,z_dim,pde_params,X_range,N)
    elif pde_type == 'Burgers':
        constraints = burgers_constraints(model,z_dim,pde_params,X_range,N) 

    return constraints       


@tf.function
def linadv_constraints(model,z_dim,pde_params,X_range,N):
    vel  = pde_params[0]
    visc = pde_params[1]
    x_   = tf.random.uniform(minval=X_range[0],maxval=X_range[1],shape=[N,1])
    t_   = tf.random.uniform(minval=X_range[2],maxval=X_range[3],shape=[N,1])
    z_   = tf.random.normal([N,z_dim])
    
    with tf.GradientTape() as g1:
        g1.watch([t_,x_])
        with tf.GradientTape(persistent=True) as g2:
            g2.watch([t_,x_])
            tx_ = tf.concat((t_,x_),axis=1)
            output_ = model([tx_,z_],training=True)
        dx = g2.gradient(output_,x_)
        dt = g2.gradient(output_,t_)
        #g2.persistent=False
    d2x = g1.gradient(dx,x_) 
    pde_pen = tf.reduce_mean(tf.square(dt + vel*dx - visc*d2x))
    del g1, g2
    return [pde_pen]    

@tf.function
def burgers_constraints(model,z_dim,pde_params,X_range,N):
    visc = pde_params[0]
    x_   = tf.random.uniform(minval=X_range[0],maxval=X_range[1],shape=[N,1])
    t_   = tf.random.uniform(minval=X_range[2],maxval=X_range[3],shape=[N,1])
    z_   = tf.random.normal([N,z_dim])
    
    with tf.GradientTape() as g1:
        g1.watch([t_,x_])
        with tf.GradientTape(persistent=True) as g2:
            g2.watch([t_,x_])
            tx_ = tf.concat((t_,x_),axis=1)
            output_ = model([tx_,z_],training=True)
            flux_   = 0.5*tf.math.square(output_)
        dx   = g2.gradient(output_,x_)
        df   = g2.gradient(flux_,x_)
        dt   = g2.gradient(output_,t_)
        #g2.persistent=False
    d2x = g1.gradient(dx,x_) 
    pde_pen = tf.reduce_mean(tf.square(dt + df - visc*d2x))
    del g1, g2

    # Periodic boundary penalty
    tx_l     = tf.concat((t_,X_range[0]*tf.ones([N,1])),axis=1)
    tx_r     = tf.concat((t_,X_range[1]*tf.ones([N,1])),axis=1)
    output_l = model([tx_l,z_],training=True)
    output_r = model([tx_r,z_],training=True)
    bnd_pen  = tf.reduce_mean(tf.square(output_l - output_r))
    
    return [pde_pen, bnd_pen]