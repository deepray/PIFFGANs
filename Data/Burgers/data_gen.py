import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns
from utils import *

#sns.set_style('darkgrid')

from config import cla
PARAMS = cla()
np.random.seed(PARAMS.seed_no)

print('\n=========================== STARTING TRAINER ============================\n')

base_params= [float(i) for i in PARAMS.base_params]

assert PARAMS.visc >= 0.0, 'viscosity must be non-negative'
assert PARAMS.Nu >= PARAMS.Nu_save, 'Nu cannot be smaller than Nu_save'

print('\n--- Creating data directory\n')
datadir = f"{PARAMS.ic_type}{PARAMS.save_suffix}" 


if not os.path.exists(datadir):
    os.makedirs(datadir)
else:
    print('    *** directory aleady exists!!')    

print(f'\n--- Saving parameters to file')
param_file = datadir + '/parameters.txt'
with open(param_file,"w") as fid:
    for pname in vars(PARAMS):
        fid.write(f"{pname} = {vars(PARAMS)[pname]}\n")      


print(f'\n--- Generating space-time mesh of size {PARAMS.Nx}X{PARAMS.Nt}')
x = np.linspace(-1, 1, PARAMS.Nx).flatten()
t = np.linspace(0,PARAMS.Tfinal,PARAMS.Nt).flatten()

# Generating samples
u_data = np.zeros((PARAMS.Nu,PARAMS.Nt,PARAMS.Nx))


print(f'\n--- Initializing {PARAMS.Nu} samples')

u_data[:,0,:]  = Initialize(Nu=PARAMS.Nu,
                            x=x,
                            ic_type=PARAMS.ic_type,
                            params=PARAMS.base_params,
                            seed=PARAMS.seed_no)  

h    = x[1]-x[0]
dt   = t[1]-t[0]
visc = PARAMS.visc

CFL  = dt*(np.max(abs(u_data[:,0,:]))/h + 2*visc/h**2)

print(f'\n--- Running ensemble solver for {PARAMS.Nu} samples with effective CFL={CFL:.4f}\n')

f_val = np.zeros((PARAMS.Nu,PARAMS.Nx))
for it in range(PARAMS.Nt-1):

    # UPDATES ASSUMING PERIODIC BC
    f_val[:,0]   = 0.25*(u_data[:,it,-2]**2 + u_data[:,it,0]**2) \
                   - 0.5*np.maximum(np.abs(u_data[:,it,-2]),np.abs(u_data[:,it,0]))*(u_data[:,it,0]-u_data[:,it,-2])
    f_val[:,1::] = 0.25*(u_data[:,it,0:-1]**2 + u_data[:,it,1::]**2) \
                   - 0.5*np.maximum(np.abs(u_data[:,it,0:-1]),np.abs(u_data[:,it,1::]))*(u_data[:,it,1::]-u_data[:,it,0:-1])

    u_data[:,it+1,0:-1] = u_data[:,it,0:-1] -dt/h*(f_val[:,1::]-f_val[:,0:-1]) \
                        +dt*visc/h**2*(u_data[:,it,1::] - 2*u_data[:,it,0:-1] + np.hstack((u_data[:,it,-2:-1],u_data[:,it,0:-2])))
    u_data[:,it+1,-1]   = u_data[:,it+1,0]					           
    print(f"    *** t={t[it+1]:.4f}", end='\r')
    
print(f'\n\n--- Evaluating statistics and creating plots')  
u_mean = np.mean(u_data,axis=0)
u_std  = np.std(u_data,axis=0)

ncol = 4
nrow = 4
fig1,axs1 = plt.subplots(nrow, ncol, dpi=100, figsize=(ncol*5,nrow*5))
ax1 = axs1.flatten()
ax_ind = 0
vmin = np.min(u_data)
vmax = np.max(u_data)
for i in range(ncol*nrow):
    axs = ax1[ax_ind]
    pcm = axs.imshow(np.flipud(u_data[i]),vmin=vmin,vmax=vmax,cmap='plasma',extent=[-1,1,0,PARAMS.Tfinal],aspect=2/PARAMS.Tfinal)
    fig1.colorbar(pcm,ax=axs)
    axs.set_title(f'Sample {i+1}')
    axs.set_xlabel('$x$')
    axs.set_ylabel('$t$')
    ax_ind +=1
fig1.tight_layout()    
fig1.savefig(f"{datadir}/u_samples.png")    
plt.close()


fig2,axs2 = plt.subplots(2, 2, dpi=100, figsize=(10,10))
ax2 = axs2.flatten()
ax_ind = 0
for t_ind in [0,int(PARAMS.Nt/3),int(2*PARAMS.Nt/3),PARAMS.Nt-1]:
    axs = ax2[ax_ind]
    for j in range(5):
        axs.plot(x,u_data[j,t_ind,:])
    axs.plot(x,u_mean[t_ind,:],'k',linewidth=2)     
    axs.set_title(f't={t[t_ind]:.3f}')
    axs.set_xlabel('$x$')
    axs.set_ylabel('$u$')  
    axs.set_ylim([vmin,vmax]) 
    ax_ind +=1 
plt.tight_layout()    
fig2.savefig(f"{datadir}/u_samples_line.png")    
plt.close()



fig3,axs3 = plt.subplots(1, 2, dpi=100,figsize=(10,5))
pcm = axs3[0].imshow(np.flipud(u_mean),vmin=np.min(u_mean),vmax=np.max(u_mean),cmap='plasma',extent=[-1,1,0,PARAMS.Tfinal],aspect=2/PARAMS.Tfinal)
axs3[0].set_title('Mean')
axs3[0].set_xlabel('$x$')
axs3[0].set_xlabel('$t$')
fig3.colorbar(pcm,ax=axs3[0])
    
pcm = axs3[1].imshow(np.flipud(u_std),vmin=np.min(u_std),vmax=np.max(u_std),cmap='plasma',extent=[-1,1,0,PARAMS.Tfinal],aspect=2/PARAMS.Tfinal)
axs3[0].set_title('Std')
axs3[1].set_xlabel('$x$')
axs3[1].set_xlabel('$t$')
fig3.colorbar(pcm,ax=axs3[1])  
fig3.tight_layout()
fig3.savefig(f"{datadir}/u_stats.png")
plt.close()  


fig4,axs4 = plt.subplots(2, 2, dpi=100, figsize=(10,10))
ax4 = axs4.flatten()
ax_ind = 0
for t_ind in [0,int(PARAMS.Nt/3),int(2*PARAMS.Nt/3),PARAMS.Nt-1]:
    axs = ax4[ax_ind]
    axs.plot(x,u_mean[t_ind,:],linewidth=2,label='Mean')  
    axs.plot(x,u_mean[t_ind,:]+u_std[t_ind,:],ls='dashed',linewidth=2,label='Mean + Std')    
    axs.plot(x,u_mean[t_ind,:]-u_std[t_ind,:],ls='dashdot',linewidth=2,label='Mean - Std') 
    axs.set_title(f't={t[t_ind]:.3f}')
    axs.set_xlabel('$x$')
    axs.set_ylabel('$u$')  
    axs.set_ylim([vmin,vmax])
    axs.legend() 
    ax_ind +=1 
plt.tight_layout()    
fig4.savefig(f"{datadir}/u_stats_line.png")    
plt.close()



print(f'\n--- Saving data')
print(f'\n    *** saving {PARAMS.Nu_save} of {PARAMS.Nu} samples')  
# Save data to file.
# NOTE: The ordering of x1 and x2 depends on the
#       way u_data is stored. The second index of
#       u_data corresponds to variation in x1
#       while the third index in x2
np.save(f"{datadir}/x1.npy",t) 
np.save(f"{datadir}/x2.npy",x)
np.save(f"{datadir}/u_data.npy",u_data[:PARAMS.Nu_save])
np.save(f"{datadir}/u_mean.npy",u_mean)
np.save(f"{datadir}/u_std.npy",u_std)


print('\n=========================== FINISHED ============================\n')




