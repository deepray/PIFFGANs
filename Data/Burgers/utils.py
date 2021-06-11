import numpy as np

def Initialize(Nu,x,ic_type,params,seed=None):

	np.random.seed(seed)

	u0   = np.zeros((Nu,len(x)))

	if ic_type == 'Jump':
		assert len(params) == 6
		ub    = params[0]
		uj    = params[1]
		x1    = params[2]
		x2    = params[3]
		u_eps = params[4]
		x_eps = params[5]

		x1_pert = x1 + x_eps*(2.0*np.random.uniform(0,1,Nu) - 1.0)
		x2_pert = x2 + x_eps*(2.0*np.random.uniform(0,1,Nu) - 1.0)
		uj_pert = uj + u_eps*(2.0*np.random.uniform(0,1,Nu) - 1.0)

		for i in range(Nu):
			u0[i,:] = ub + (uj_pert[i]-ub)*(x>x1_pert[i])*(x<x2_pert[i])


	elif ic_type == 'SineExp':
		assert len(params) == 4
		um    = params[0]
		x0    = params[1]
		u_eps = params[2]
		x_eps = params[3]

		x0_pert = x0 + x_eps*(2.0*np.random.uniform(0,1,Nu) - 1.0)
		um_pert = um + u_eps*(2.0*np.random.uniform(0,1,Nu) - 1.0)

		for i in range(Nu):
			u0[i,:] = um_pert[i]*np.sin(4*np.pi*(x-x0_pert[i]))*(np.exp(-x*x/0.1))

	return u0




