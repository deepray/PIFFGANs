import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import tensorflow as tf
from tensorflow import keras

np.random.seed(1008)

pi         = np.pi
l2_param   = 0.001
Act_func   = 'sine'
Act_param  = 0.2
use_bias   = True


if Act_func == 'ReLU':
  activation_func = keras.layers.ReLU(negative_slope=Act_param)
elif Act_func == 'tanh':  
  activation_func = keras.layers.Activation('tanh')
elif Act_func == 'sigmoid':  
  activation_func = keras.layers.Activation('sigmoid')
elif Act_func == 'sine':  
  activation_func = tf.math.sin 


# Create generator (standard)
def generator_std(X_dim,output_dim,Z_dim=1,width=15,depth=5):

    assert depth > 1, 'Depth of generator must be greater than 1'
    
    input_X = keras.Input(shape=(X_dim))
    input_Z = keras.Input(shape=(Z_dim))
    input_XZ = keras.layers.Concatenate()([input_X,input_Z])

    X = input_XZ

    # (depth -1) Hidden layers
    for d in range(depth-1):
        X = keras.layers.Dense(units=width, 
                               activation=activation_func, 
                               use_bias=use_bias,
                               kernel_regularizer=keras.regularizers.l2(l2_param)
                               )(X)
        X = keras.layers.BatchNormalization()(X)                       

    # Output layer
    Y = keras.layers.Dense(units=output_dim, 
                           activation=None, 
                           use_bias=use_bias,
                           kernel_regularizer=keras.regularizers.l2(l2_param)
                           )(X)


    
    model = keras.Model(inputs=[input_X, input_Z], outputs=Y)

    return model

# Create generator (FF)
def generator_FF(X_dim,output_dim,B,Z_dim=1,width=15,depth=5):

    assert depth > 1, 'Depth of generator must be greater than 1'
    
    input_X = keras.Input(shape=(X_dim))
    input_Z = keras.Input(shape=(Z_dim))

    BX      = tf.linalg.matmul(a=input_X,b=B,transpose_b=True)
    FFX1    = tf.math.cos(2*pi*BX)
    FFX2    = tf.math.sin(2*pi*BX)
    input_FFXZ = keras.layers.Concatenate()([FFX1,FFX2,input_Z])

    X = input_FFXZ

    # (depth -1) Hidden layers
    for d in range(depth-1):
        X = keras.layers.Dense(units=width, 
                               activation=activation_func, 
                               use_bias=use_bias,
                               kernel_regularizer=keras.regularizers.l2(l2_param)
                               )(X)
        X = keras.layers.BatchNormalization()(X)                       


    # Output layer
    Y = keras.layers.Dense(units=output_dim, 
                           activation=None, 
                           use_bias=use_bias,
                           kernel_regularizer=keras.regularizers.l2(l2_param)
                           )(X)
   
    model = keras.Model(inputs=[input_X, input_Z], outputs=Y)

    return model    

# Create critic (MLP)
def critic_MLP(input_dim,width=15,depth=5):

    assert depth > 1, 'Depth of generator must be greater than 1'
    
    input_X = keras.Input(shape=(input_dim))

    X = input_X

    # (depth -1) Hidden layers
    for d in range(depth-1):
        X = keras.layers.Dense(units=width, 
                               activation=activation_func, 
                               use_bias=use_bias,
                               kernel_regularizer=keras.regularizers.l2(l2_param)
                               )(X)
        X = keras.layers.LayerNormalization()(X)                       

    # Output layer
    Y = keras.layers.Dense(units=1, 
                           activation=None, 
                           use_bias=use_bias,
                           kernel_regularizer=keras.regularizers.l2(l2_param)
                           )(X)


    
    model = keras.Model(inputs=input_X, outputs=Y)

    return model   

# Create critic (CNN)
def critic_CNN(input_W,input_H,k0=16):

    input_X = keras.Input(shape=(input_H,input_W,1))

    X = input_X

    # CNN + downsampling layers
    for i in range(3):
        X = keras.layers.Conv2D(filters=k0*(2**i), 
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                activation=activation_func, 
                                use_bias=use_bias,
                                kernel_regularizer=keras.regularizers.l2(l2_param)
                                )(X)
        X = keras.layers.LayerNormalization()(X)                        
        X = keras.layers.AveragePooling2D(pool_size=2,strides=2)(X)                        

    # FC layers
    X = tf.keras.layers.Flatten()(X)
    X = keras.layers.Dense(units=32, 
                           activation=activation_func, 
                           use_bias=use_bias,
                           kernel_regularizer=keras.regularizers.l2(l2_param)
                           )(X)
    X = keras.layers.LayerNormalization()(X)
    
    # Output layer                       
    Y = keras.layers.Dense(units=1, 
                           activation=None, 
                           use_bias=use_bias,
                           kernel_regularizer=keras.regularizers.l2(l2_param)
                           )(X)                       

    
    model = keras.Model(inputs=input_X, outputs=Y)

    return model     
    
@tf.function
def gradient_penalty(real, fake, model, k=10, p=2):
    shape   = tf.concat((tf.shape(real)[0:1], tf.tile([1], [real.shape.ndims - 1])), axis=0)
    epsilon = tf.random.uniform(shape, 0.0, 1.0)
    x_hat   = epsilon * real + (1 - epsilon) * fake
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat   = model(x_hat, training=True)   
    gradients = t.gradient(d_hat, x_hat)
    ddx = tf.reduce_sum(1e-8 + tf.square(gradients), axis=tf.range(1,real.shape.ndims))
    d_regularizer = k*tf.reduce_mean(tf.pow(ddx,p))
    return d_regularizer 
  
