from svr_object import *

def svr_trainer(data, label, c, eps):

	svrobj = SVR(data)
	ntrain = len(data)
	alpha0 = np.zeros((ntrain,1))

	M = np.zeros(ntrain,ntrain)
	#Setting up gram matrix
	for i in range(ntrain):
		for j in range(i+1):
			M[i][j] = svrobj.kernel_function(data[i], data[j])
			M[j][i] = M[i][j]

	M = M + (1/c)*np.eye(ntrain)