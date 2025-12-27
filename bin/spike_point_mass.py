import numpy as np
import scipy as sp
import math
import pandas as pd 
import time
import sys
import geweke
import os
import gc

def sample_pie(gamma,pie_a,pie_b):
	a_new = np.sum(gamma)+pie_a
	b_new = np.sum(1-gamma)+pie_b
	pie_new = np.random.beta(a_new,b_new)
	return(pie_new)

def sample_sigma_1(beta,gamma,a_sigma,b_sigma):
	a_new = 0.5*np.sum(gamma)+a_sigma
	b_new = 0.5*np.sum(np.multiply(np.square(beta),gamma))+b_sigma
	sigma_1_neg2 =np.random.gamma(a_new,1.0/b_new)
	sigma_1_new = math.sqrt(1/sigma_1_neg2)
	return(sigma_1_new)

def sample_sigma_e(y,H_beta,C_alpha,a_e,b_e):
	n = len(y)
	a_new = float(n)/2+a_e
	resid = y - H_beta - C_alpha
	b_new = np.sum(np.square(resid))/2+b_e
	sigma_e_neg2 =np.random.gamma(a_new,1.0/b_new)
	sigma_e_new = math.sqrt(1/sigma_e_neg2)
	return(sigma_e_new)

def sample_alpha(y,H_beta,C,alpha,sigma_e,C_alpha):

	r,c = C.shape
	if c == 1:
		new_variance = 1/(np.linalg.norm(C[:,0])**2*sigma_e**-2)
		new_mean = new_variance*np.dot((y-H_beta),C[:,0])*sigma_e**-2
		alpha = np.random.normal(new_mean,math.sqrt(new_variance))
		C_alpha = C[:,0] * alpha 
	else:
		for i in range(c):
			new_variance = 1/(np.sum(C[:,i]**2)*sigma_e**-2)
			C_alpha_negi = C_alpha - C[:,i] * alpha[i]
			new_mean = new_variance*np.dot(y-C_alpha_negi-H_beta,C[:,i])*sigma_e**-2
			alpha[i] = np.random.normal(new_mean,math.sqrt(new_variance))
			C_alpha = C_alpha_negi + C[:,i] * alpha[i]
	return(alpha,C_alpha)


def sample_gamma(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta):
	sigma_e_neg2 = sigma_e**-2
	sigma_1_neg2 = sigma_1**-2
	e = y - C_alpha -  H_beta 
	residual_negi = np.transpose(np.ones((len(beta),len(y))) * e)+ H * beta
	variance = 1/(np.sum(H**2,axis=0) * sigma_e_neg2+sigma_1_neg2)
	mean = variance * np.sum(residual_negi * H,axis=0) * sigma_e_neg2
	f = np.sqrt(1/(np.sum(H**2,axis=0) * sigma_1**2 * sigma_e_neg2 + 1))
	gamma_0_pie = (1-pie) / ((1-pie)+pie*(f*np.exp(0.5*mean**2/variance)))	
	gamma = np.random.binomial(1,1-gamma_0_pie)
	return(gamma)

def sample_beta(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta):

	sigma_e_neg2 = sigma_e**-2
	sigma_1_neg2 = sigma_1**-2	
	for j in range(len(beta)):
		if gamma[j] == 0:
			H_beta = H_beta - H[:,j] * beta[j]
			beta[j] = 0
		else:
			H_beta_negj = H_beta - H[:,j] * beta[j]
			new_variance = 1/(np.sum(H[:,j]**2)*sigma_e_neg2+sigma_1_neg2)
			residual = y - C_alpha -  H_beta + H[:,j] * beta[j]
			new_mean = new_variance*np.dot(residual,H[:,j])*sigma_e_neg2
			beta[j] = np.random.normal(new_mean,math.sqrt(new_variance))
			if abs(beta[j]) < 0.05:
				beta[j] = 0
			H_beta = H_beta_negj + H[:,j] * beta[j]
	return(beta,H_beta)

def sampling(verbose,y,C,HapDM,iters,prefix,num,trace_container,gamma_container,beta_container,alpha_container,pi_b):

	## set random seed for the process
	np.random.seed(int(time.time()) + os.getpid())

	#initiate beta,gamma and H matrix
	C_r, C_c = C.shape

	H = np.array(HapDM)

	H_r,H_c = H.shape
	##specify hyper parameters
	pie_a = 1
	pie_b = H_c *pi_b
	a_sigma = 1
	b_sigma = 1
	a_e = 1
	b_e = 1

	sigma_1 = math.sqrt(1/np.random.gamma(a_sigma,b_sigma))
	sigma_e = math.sqrt(1/np.random.gamma(a_e,b_e))
	pie = np.random.beta(pie_a,pie_b)
	
	#print("initiate:",sigma_1,sigma_e,pie)

	#initiate beta,gamma and H matrix
	C_r, C_c = C.shape

	H = np.array(HapDM)

	#for simulation only
	H_r,H_c = H.shape

	#initiate alpha, alpha_trace, beta_trace and gamma_trace

	it = 0
	burn_in_iter = 2000
	trace = np.empty((iters-2000,5))
	alpha_trace = np.empty((iters-2000,C_c))
	gamma_trace = np.empty((iters-2000,H_c))
	beta_trace = np.empty((iters-2000,H_c))
	top5_beta_trace = np.empty((iters-2000,5))

	alpha = np.random.random(size = C_c)
	gamma = np.random.binomial(1,pie,H_c)
	beta = np.array(np.zeros(H_c))

	for i in range(H_c):
		if gamma[i] == 0:
			beta[i] = 0
		else:
			beta[i] = np.random.normal(0,sigma_1) 

	#start sampling

	H_beta = np.matmul(H,beta)
	C_alpha = np.matmul(C,alpha)

	while it < iters:
		before = time.time()
		sigma_1 = sample_sigma_1(beta,gamma,a_sigma,b_sigma)
		pie = sample_pie(gamma,pie_a,pie_b)
		sigma_e = sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
		gamma = sample_gamma(y,C_alpha,H,beta,pie,sigma_1,sigma_e,gamma,H_beta)
		alpha,C_alpha = sample_alpha(y,H_beta,C,alpha,sigma_e,C_alpha)
		beta,H_beta = sample_beta(y,C_alpha,H,beta,gamma,sigma_1,sigma_e,H_beta)
		genetic_var = np.var(H_beta)
		pheno_var = np.var(y)
		large_beta_ratio = np.sum(np.absolute(beta) > 0.3) / H_c
		total_heritability = genetic_var / pheno_var
		after = time.time()
		if it > 100 and total_heritability > 1:
			#print("unrealistic beta sample",it,pie,sigma_1,sigma_e,sum(gamma),large_beta_ratio,total_heritability)
			continue

		else:
			if verbose:
				print(num,it,str(after - before),sigma_1,sigma_e,large_beta_ratio,total_heritability,sum(gamma))

			if it >= burn_in_iter:
				trace[it-burn_in_iter,:] = [sigma_1,sigma_e,large_beta_ratio,total_heritability,sum(gamma)]
				gamma_trace[it-burn_in_iter,:] = gamma
				beta_trace[it-burn_in_iter,:] = beta
				alpha_trace[it-burn_in_iter,:] = alpha
				top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

			if it >= burn_in_iter + 9999: # after burn-in iterations, test convergence

				max_z = []
				
				for a in range(C_c):
					after_burnin_alpha = alpha_trace[:,a]
					alpha_zscores = geweke.geweke(after_burnin_alpha)[:,1]
					max_z.append(np.amax(np.absolute(alpha_zscores)))

				for b in range(5):
					after_burnin_beta = top5_beta_trace[:,b]
					beta_zscores = geweke.geweke(after_burnin_beta)[:,1]
					max_z.append(np.amax(np.absolute(beta_zscores)))

				#convergence for large beta ratio
				after_burnin_pie = trace[:,2]
				pie_zscores = geweke.geweke(after_burnin_pie)[:,1]
				max_z.append(np.amax(np.absolute(pie_zscores)))

				#convergence for total heritability
				after_burnin_var = trace[:,3]
				var_zscores = geweke.geweke(after_burnin_var)[:,1]
				max_z.append(np.amax(np.absolute(var_zscores)))

				# #convergence for sigma_1
				# after_burnin_sigma1 = trace[:,0]
				# sigma1_zscores = geweke.geweke(after_burnin_sigma1)[:,1]
				# max_z.append(np.amax(np.absolute(sigma1_zscores)))

				#convergence for sigma_e
				after_burnin_sigmae = trace[:,1]
				sigmae_zscores = geweke.geweke(after_burnin_sigmae)[:,1]
				max_z.append(np.amax(np.absolute(sigmae_zscores)))
				
				if  np.amax(max_z) < 1.5:
					print("Chain %i: convergence has been reached at %i iterations." %(num,it))
					break

				else:
					trace_ = np.empty((1000,5))
					gamma_trace_ = np.empty((1000,H_c))
					beta_trace_ = np.empty((1000,H_c))
					alpha_trace_ = np.empty((1000,C_c))
					top5_beta_trace_ = np.empty((1000,5))

					trace = np.concatenate((trace[-(iters - burn_in_iter-1000):,:],trace_),axis=0)
					gamma_trace = np.concatenate((gamma_trace[-(iters - burn_in_iter-1000):,:],gamma_trace_),axis=0)
					beta_trace = np.concatenate((beta_trace[-(iters - burn_in_iter-1000):,:],beta_trace_),axis=0)
					alpha_trace = np.concatenate((alpha_trace[-(iters - burn_in_iter-1000):,:],alpha_trace_),axis=0)
					top5_beta_trace = np.concatenate((top5_beta_trace[-(iters - burn_in_iter-1000):,:],top5_beta_trace_),axis = 0)

					burn_in_iter += 1000
					iters += 1000

			if (it - burn_in_iter) >= 0 and (it - burn_in_iter ) % 1000 == 0 :
				print("Chain %i has sampled %i iterations " %(num,it), str(after - before),trace[it-burn_in_iter,:])

			it += 1
	
	# trace_container[num] = trace
	# alpha_container[num] = alpha_trace
	# beta_container[num] = beta_trace
	# gamma_container[num] = gamma_trace

	# trace values
	trace_container[num] = {'avg': np.mean(trace,axis=0),
							'sd' : np.std(trace,axis=0)}
	del trace
	gc.collect()

	#print("trace_container[num]",num,trace_container[num]['avg'])
	#alpha values
	alpha_container[num] = {'avg': np.mean(alpha_trace,axis=0),
							'sd': np.std(alpha_trace,axis=0)}
	del alpha_trace
	gc.collect()
	#print("alpha_container[num]",num,alpha_container[num]['avg'])
	#beta values

	beta_container[num] = {'avg':np.mean(beta_trace,axis=0),
							'sd':np.std(beta_trace,axis=0)}
	del beta_trace
	gc.collect()

	gamma_container[num] = {'kmer':np.mean(gamma_trace,axis=0)}
	del gamma_trace
	gc.collect()




