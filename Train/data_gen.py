import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from config import cla
#sns.set_style('darkgrid')

PARAMS = cla()
np.random.seed(PARAMS.seed_no)

#============== Dataset ========================
class CreateDataset:
    def __init__(self,save_dir):
        x1_data = np.load(f"../Data/{PARAMS.data_dir}/x1.npy").flatten().astype(np.float32)
        x2_data = np.load(f"../Data/{PARAMS.data_dir}/x2.npy").flatten().astype(np.float32)
        u_data  = np.load(f"../Data/{PARAMS.data_dir}/u_data.npy").astype(np.float32)
        u_mean  = np.load(f"../Data/{PARAMS.data_dir}/u_mean.npy").astype(np.float32)
        u_std   = np.load(f"../Data/{PARAMS.data_dir}/u_std.npy").astype(np.float32)
        
        # Full resolution
        Nx1_all = x1_data.shape[0]
        x1L     = x1_data[0]
        x1R     = x1_data[-1]
        Nx2_all = x2_data.shape[0]
        x2L     = x2_data[0]
        x2R     = x2_data[-1]

        # Creating probe points
        x1_sk  = int(np.ceil(1/PARAMS.Nx1_frac))
        x1     = x1_data[0::x1_sk]
        x1_ind = np.arange(0,Nx1_all,x1_sk)
        if x1_ind[-1] != Nx1_all:
            x1     = np.append(x1,x1_data[-1])
            x1_ind = np.append(x1_ind,Nx1_all-1)
        Nx1 = x1.shape[0]

        x2_sk  = int(np.ceil(1/PARAMS.Nx2_frac))
        x2     = x2_data[0::x2_sk]
        x2_ind = np.arange(0,Nx2_all,x2_sk)
        if x2_ind[-1] != Nx2_all:
            x2     = np.append(x2,x2_data[-1])
            x2_ind = np.append(x2_ind,Nx2_all-1)
        Nx2 = x2.shape[0]  

        # Extract data at probes
        data = u_data[0:PARAMS.Nu,:,:][:,x1_ind,:][:,:,x2_ind]           

        self.save_dir  = save_dir
        self.x1_data   = x1_data
        self.x2_data   = x2_data
        self.x1        = x1 
        self.x2        = x2 
        self.Nx1       = Nx1 
        self.Nx2       = Nx2
        self.Nx1_all   = Nx1_all 
        self.Nx2_all   = Nx2_all
        self.dataset   = tf.data.Dataset.from_tensor_slices(data)
        self.u_mean    = u_mean
        self.u_std     = u_std

        plot_samples(data,x1,x2,save_dir,'true_samples')

 
def plot_samples(data,x1,x2,save_dir,fname):

    x1L,x1R,x2L,x2R = x1[0],x1[-1],x2[0],x2[-1]
    Nx1  = x1.shape[0]
    Nx2  = x2.shape[0]
    nrow = 3
    ncol = 3
    fig1,axs1 = plt.subplots(nrow, ncol, dpi=100, figsize=(ncol*5,nrow*5))
    ax1 = axs1.flatten()
    ax_ind = 0
    vmin = np.min(data[:nrow*ncol])
    vmax = np.max(data[:nrow*ncol])
    for i in range(ncol*nrow):
        axs = ax1[ax_ind]
        pcm = axs.imshow(np.flipud(data[i]),vmin=vmin,vmax=vmax,cmap='plasma',extent=[x2L,x2R,x1L,x1R],aspect=(x2R-x2L)/(x1R-x1L))
        fig1.colorbar(pcm,ax=axs)
        axs.set_title(f'Sample {i+1}')
        axs.set_xlabel('$x_1$')
        axs.set_ylabel('$x_2$')
        ax_ind +=1
    fig1.tight_layout()    
    fig1.savefig(f"{save_dir}/{fname}.png")    
    plt.close()

    with sns.axes_style("darkgrid"):
        fig2,axs2 = plt.subplots(2, 2, dpi=100, figsize=(10,10))
        ax2 = axs2.flatten()
        ax_ind = 0
        for x1_ind in [0,int(Nx1/3),int(2*Nx1/3),Nx1-1]:
            axs = ax2[ax_ind]
            for j in range(nrow*ncol):
                axs.plot(x2,data[j,x1_ind,:])  
            axs.set_title(f'x_1={x1[x1_ind]:.3f}')
            axs.set_xlabel('$x_2$')
            axs.set_ylabel('$u$')  
            axs.set_ylim([vmin,vmax]) 
            ax_ind +=1 
        plt.tight_layout()    
        fig2.savefig(f"{save_dir}/{fname}_lines.png")    
        plt.close()       


def plot_stats(mean,std,ref_mean,ref_std,x1,x2,save_dir,fname):

    x1L,x1R,x2L,x2R = x1[0],x1[-1],x2[0],x2[-1]
    Nx1  = x1.shape[0]
    Nx2  = x2.shape[0]
    mean_min, mean_max = np.min(ref_mean),np.max(ref_mean)
    std_min, std_max   = np.min(ref_std),np.max(ref_std)

    fig1,axs1 = plt.subplots(2, 2, dpi=100, figsize=(10,10))
    
    axs = axs1[0][0]
    pcm = axs.imshow(np.flipud(mean),vmin=mean_min,vmax=mean_max,cmap='plasma',extent=[x2L,x2R,x1L,x1R],aspect=(x2R-x2L)/(x1R-x1L))
    fig1.colorbar(pcm,ax=axs)
    axs.set_title(f'Mean')
    axs.set_xlabel('$x_1$')
    axs.set_ylabel('$x_2$')

    axs = axs1[0][1]
    pcm = axs.imshow(np.flipud(std),vmin=std_min,vmax=std_max,cmap='plasma',extent=[x2L,x2R,x1L,x1R],aspect=(x2R-x2L)/(x1R-x1L))
    fig1.colorbar(pcm,ax=axs)
    axs.set_title(f'Std')
    axs.set_xlabel('$x_1$')
    axs.set_ylabel('$x_2$')

    axs = axs1[1][0]
    mean_diff = np.abs(mean-ref_mean)
    pcm = axs.imshow(np.flipud(mean_diff),vmin=np.min(mean_diff),vmax=np.max(mean_diff),cmap='plasma',extent=[x2L,x2R,x1L,x1R],aspect=(x2R-x2L)/(x1R-x1L))
    fig1.colorbar(pcm,ax=axs)
    axs.set_title('Error in mean')
    #axs.set_title(f'Error in Mean [{np.min(mean_diff)},{np.max(mean_diff)}]')
    axs.set_xlabel('$x_1$')
    axs.set_ylabel('$x_2$')

    axs = axs1[1][1]
    std_diff = np.abs(std-ref_std)
    pcm = axs.imshow(np.flipud(std_diff),vmin=np.min(std_diff),vmax=np.max(std_diff),cmap='plasma',extent=[x2L,x2R,x1L,x1R],aspect=(x2R-x2L)/(x1R-x1L))
    fig1.colorbar(pcm,ax=axs)
    axs.set_title(f'Error in Std')
    axs.set_xlabel('$x_1$')
    axs.set_ylabel('$x_2$')
        
    fig1.tight_layout()    
    fig1.savefig(f"{save_dir}/{fname}.png")    
    plt.close()

    nrow = 3
    ncol = 3
    with sns.axes_style("darkgrid"):
        fig2,axs2 = plt.subplots(2, 2, dpi=100, figsize=(10,10))
        ax2 = axs2.flatten()
        ax_ind = 0
        for x1_ind in [0,int(Nx1/3),int(2*Nx1/3),Nx1-1]:
            axs = ax2[ax_ind]
            axs.plot(x2,mean[x1_ind,:],linewidth=2,label='Predicted mean',color='tab:orange')  
            axs.plot(x2,mean[x1_ind,:]+std[x1_ind,:],ls='dashed',linewidth=2,color='tab:orange') 
            axs.plot(x2,mean[x1_ind,:]-std[x1_ind,:],ls='dashdot',linewidth=2,color='tab:orange') 
            axs.plot(x2,ref_mean[x1_ind,:],linewidth=2,label='Ref. mean',color='tab:blue')  
            axs.plot(x2,ref_mean[x1_ind,:]+ref_std[x1_ind,:],ls='dashed',linewidth=2,color='tab:blue') 
            axs.plot(x2,ref_mean[x1_ind,:]-ref_std[x1_ind,:],ls='dashdot',linewidth=2,color='tab:blue') 
            axs.set_title(f'x_1={x1[x1_ind]:.3f}')
            axs.set_xlabel('$x_2$')
            axs.set_ylabel('$u$')  
            axs.set_ylim([mean_min-std_max,mean_max + std_max]) 
            axs.legend()
            ax_ind +=1 
        plt.tight_layout()    
        fig2.savefig(f"{save_dir}/{fname}_lines.png")    
        plt.close() 




