# Physics-informed Fourier Feature GANs (PIFFGANs)
### Author: Deep Ray, USC
### Date: June 6, 2021

This python script is written to train WGANS with PDE-based constraints. 

## Setting up a virtual environment 
These are the instructions to set up a virtual environment using ``virtualenv`` or ``anaconda`` and install the following packages:

* ``tensorflow`` (or ``tensorflow-gpu`` if you plan to use a GPU)
* ``matplotlib``
* ``pandas``
* ``seaborn``

## The setup
Currently, the code is setup for a two-dimensional parametrized PDE. See ["discription"](discription.html) for details of the models used.

Currently, the code can make use of the following PDEs (set using the ``--pde_type`` parameter)

* 'None': No PDE is used, the data is treated as images.
* 'LinAdv': Viscous linear advection. 
* 'Burgers': Viscous Burgers equation.

The PDE residual constraints are described in ``Train/pde_utils.py``, which returns a list of residuals associated with the PDE.


## Create you own data
To create your own dataset (from a PDE or otherwise), you need to create the following Numpy arrays and save it to a directory (preferably a sub-directory in ``Data``)

* ``x1.npy``: Array of size $$N_1$$, containing the node locations in the $$x_1$$ coordinate directions.
* ``x2.npy``: Array of size $$N_2$$, containing the node locations in the $$x_2$$ coordinate directions.
* ``u_data.npy``: Array of size $$N \times N_1 \times N_2$$, where $$N$$ is the number of samples.
* ``u_mean.npy``: Array of size $$N_1 \times N_2$$ containing the sample mean at each node in the grid.
* ``u_std.npy``: Array of size $$N_1 \times N_2$$ containing the sample standard deviation at each node in the grid.

If the you plan to use a new type of PDE, then after creating the above numpy arrays

* Add a new PDE type choice to the ``--pde_type`` argument in ``Train/config.py``
* Add the list of PDE constraints for this new PDE in ``Train/pde_utils.py``. Don't foget to augment the pde type to the function ``pde_constraints``.

## Try a quick test
Activate your virtual environment. First create a sample data set by running the following from within the ``Data`` directory:


and try running the following from within the ``Train/Burgers`` directory:
~~~
python3 trainer.py \
        --Nx1_frac=128 \
        --Nx2_frac=128 \
        --Tfinal=0.2 \
        --Nu=10000 \
        --Nu_save=1000 \
        --ictype=SineExp \
        --base_params 0.8 0.0 0.1 0.1
        --visc=0.01
~~~

Have a look at the data file created in ``Train/Burgers/SineExp``. Next, go to the ``Train`` directory and run the following command to train a network

~~~
python3 trainer.py \
        --data_dir=../Data/Burgers/SineExp \
        --Nx1_frac=0.1 \
        --Nx2_frac=0.1 \
        --g_type='FF' \
        --g_width=100 \
        --d_type='CNN' \
        --d_width=100 \
        --n_epoch=200 \
        --pde_type=Burgers \
        --z_dim=5 \
        --pde_params 0.0 \
        --pdecon_coef 0.0 \
        --batch_size=50 \
        --savefig_freq=10 \
        --Nu=200 \
        --save_dir=Results
~~~

The training results and checkpoints will be saved in ``xps/Results``



