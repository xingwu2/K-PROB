
import numpy as np
import scipy as sp
import math
import pandas as pd 
import time
import geweke
import os
import gc
import sys
from scipy.sparse import csc_matrix
from numba import njit

import utility_functions as uf

@njit
def sample_gamma_numba(beta, sigma_0, sigma_1, pie, gamma_out):
	log_prior_ratio = math.log(pie) - math.log1p(-pie)
	log_scale_ratio = math.log(sigma_0 / sigma_1)
	precision_diff = 0.5 * (1.0/sigma_0**2 - 1.0/sigma_1**2)
    
	for i in range(beta.shape[0]):
		log_ratio = log_prior_ratio + log_scale_ratio + beta[i]**2 * precision_diff
        # stable sigmoid
		if log_ratio >= 0:
			p = 1.0 / (1.0 + math.exp(-log_ratio))
		else:
			ex = math.exp(log_ratio)
			p = ex / (1.0 + ex)
		gamma_out[i] = 1 if np.random.random() < p else 0
	return(gamma_out)


def sample_gamma(beta,sigma_0,sigma_1,pie):
	p = np.empty(len(beta))
	d1 = pie*sp.stats.norm.pdf(beta,loc=0,scale=sigma_1)
	d0 = (1-pie)*sp.stats.norm.pdf(beta,loc=0,scale=sigma_0)
	p = d1/(d0+d1)
	gamma = np.random.binomial(1,p).astype(np.uint8)
	return(gamma)

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

def sample_alpha(y,H_beta,C_alpha,C,alpha,sigma_e,C_norm_2):

	r,c = C.shape

	if c == 1:
		#new_variance = 1/(np.linalg.norm(C[:,0])**2*sigma_e**-2)
		new_variance = 1/(C_norm_2[0]*sigma_e**-2)
		new_mean = new_variance*np.dot((y-H_beta),C[:,0])*sigma_e**-2
		alpha[0] = np.random.normal(new_mean,math.sqrt(new_variance))
		C_alpha = C[:,0] * alpha[0]
	else:
		for i in range(c):
			#new_variance = 1/(np.linalg.norm(C[:,i])**2*sigma_e**-2)
			new_variance = 1/(C_norm_2[i]*sigma_e**-2)
			C_alpha_negi = C_alpha - C[:,i] * alpha[i]
			new_mean = new_variance*np.dot(y-C_alpha_negi-H_beta,C[:,i])*sigma_e**-2
			alpha[i] = np.random.normal(new_mean,math.sqrt(new_variance))
			C_alpha = C_alpha_negi + C[:,i] * alpha[i]

	return(alpha,C_alpha)

@njit
def sample_alpha_numba(y, H_beta, C_alpha, C, alpha, sigma_e, C_norm_2):
	sigma_e_neg2 = sigma_e ** -2
	ncols = alpha.shape[0]
	nrows = y.shape[0]
    
	for i in range(ncols):
		old_alpha_i = alpha[i]
        
        # Compute dot product against the "stale" residual
        # (C_alpha still contains old_alpha_i's contribution)
		dot_val = 0.0
		for r in range(nrows):
			residual = y[r] - C_alpha[r] - H_beta[r]
			dot_val += C[r, i] * residual
        
		adjusted_dot = dot_val + C_norm_2[i] * old_alpha_i
        
		new_variance = 1.0 / (C_norm_2[i] * sigma_e_neg2)
		new_mean = new_variance * sigma_e_neg2 * adjusted_dot
        
		new_alpha_i = new_mean + math.sqrt(new_variance) * np.random.randn()
		alpha[i] = new_alpha_i
        
        # Apply the net change to C_alpha in one pass
		diff = new_alpha_i - old_alpha_i
		for r in range(nrows):
			C_alpha[r] += C[r, i] * diff
    
	return (alpha, C_alpha)


def sample_beta(y,C_alpha,H_beta,H,beta,gamma,sigma_0,sigma_1,sigma_e,H_norm_2):

	sigma_e_neg2 = sigma_e**-2
	sigma_0_neg2 = sigma_0**-2
	sigma_1_neg2 = sigma_1**-2

	for i in range(len(beta)):

		H_beta_negi = H_beta - H[:,i] * beta[i] 	### original


		residual = y - C_alpha -  H_beta + H[:,i] * beta[i]		### original

		new_variance = 1/(H_norm_2[i]*sigma_e_neg2+(1-gamma[i])*sigma_0_neg2+gamma[i]*sigma_1_neg2)		### original


		new_mean = new_variance*np.dot(residual,H[:,i])*sigma_e_neg2		### original

		beta[i] = np.random.normal(new_mean,math.sqrt(new_variance))		### original

		H_beta = H_beta_negi + H[:,i] * beta[i]		### original


	return(beta,H_beta)



@njit
def sample_beta_numba(y, C_alpha, H_beta, H, beta, gamma, sigma_0, sigma_1, sigma_e, H_norm_2):
	sigma_e_neg2 = sigma_e ** -2
	sigma_0_neg2 = sigma_0 ** -2
	sigma_1_neg2 = sigma_1 ** -2
	ncols = beta.shape[0]
	nrows = y.shape[0]
    
	for i in range(ncols):

		for r in range(nrows):
			H_beta[r] -= H[r, i] * beta[i]

        # Compute the dot product over the column using the updated H_beta.
		dot_val = 0.0
		for r in range(nrows):
            # residual = y[r] - C_alpha[r] - H_beta[r]
			res_val = y[r] - C_alpha[r] - H_beta[r] 
			dot_val += res_val * H[r, i]
        
		new_variance = 1.0 / (H_norm_2[i]*sigma_e_neg2 + (1 - gamma[i])*sigma_0_neg2 + gamma[i]*sigma_1_neg2)
		new_mean = new_variance * sigma_e_neg2 * dot_val
        
        # Sample new beta using standard normal (Numba supports np.random.randn)
		beta[i] = new_mean + math.sqrt(new_variance) * np.random.randn()
       
        # Update H_beta with the new contribution.
		for r in range(nrows):
			H_beta[r] += H[r, i] * beta[i]
    
	return (beta, H_beta)

@njit(parallel=False,fastmath=True)
def sample_beta_numba_optimized(y, C_alpha, H_beta, H, beta, gamma, sigma_0, sigma_1, sigma_e, H_norm_2):
	sigma_e_neg2 = sigma_e ** -2
	sigma_0_neg2 = sigma_0 ** -2
	sigma_1_neg2 = sigma_1 ** -2
	ncols = beta.shape[0]
	nrows = y.shape[0]
	y_minus_C = y - C_alpha
    
	for i in range(ncols):
		H_col = H[:, i]

		H_beta -= H_col * beta[i]

        # Compute the dot product over the column using the updated H_beta.
		dot_val = np.dot(y_minus_C - H_beta, H_col)
        
		prec = sigma_1_neg2 if gamma[i] == 1 else sigma_0_neg2
		new_variance = 1.0 / (H_norm_2[i]*sigma_e_neg2 + prec)
		new_mean = new_variance * sigma_e_neg2 * dot_val
        
        # Sample new beta using standard normal (Numba supports np.random.randn)
		beta[i] = new_mean + math.sqrt(new_variance) * np.random.randn()
       
        # Update H_beta with the new contribution.
		H_beta += H_col * beta[i]
    
	return (beta, H_beta)

# @njit
# def sample_beta_numba_optimized(y, C_alpha, H_beta, H, beta, gamma, sigma_0, sigma_1, sigma_e, H_norm_2):
# 	sigma_e_neg2 = sigma_e ** -2
# 	sigma_0_neg2 = sigma_0 ** -2
# 	sigma_1_neg2 = sigma_1 ** -2
# 	ncols = beta.shape[0]
# 	nrows = y.shape[0]
    
# 	for i in range(ncols):
# 		old_beta_i = beta[i]

# 		dot_val = 0.0
# 		for r in range(nrows):
# 			residual = y[r] - C_alpha[r] - H_beta[r]
# 			dot_val += H[r, i] * residual
        
# 		adjusted_dot = dot_val + H_norm_2[i] * old_beta_i
        
# 		prec_selection = sigma_1_neg2 if gamma[i] == 1 else sigma_0_neg2
# 		new_var = 1.0 / (H_norm_2[i] * sigma_e_neg2 + prec_selection)
# 		new_mean = new_var * sigma_e_neg2 * adjusted_dot
        
# 		new_beta_i = new_mean + math.sqrt(new_var) * np.random.randn()
# 		beta[i] = new_beta_i
        
# 		diff = new_beta_i - old_beta_i
# 		for r in range(nrows):
# 			H_beta[r] += H[r, i] * diff
# 	return (beta, H_beta)



def sampling(verbose,y,C,H,sig0_initiate,prefix,num,trace_container,gamma_container,beta_container,alpha_container,convergence_container,pi_b):

	## set random seed for the process
	np.random.seed(int(time.time()) + os.getpid())

	H_r,H_c = H.shape
	C_c = C.shape[1]

	##specify hyper parameters
	pie_a = 1.0
	#pie_b = 99.0
	pie_b = pie_a / pi_b - 	pie_a
	a_sigma = 1.0
	b_sigma = 1.0
	a_e = 1.0
	b_e = 1.0
	
	H_var = np.sum(np.var(H,axis=0)).astype(np.float32)
	sigma_0 = np.float32(np.sqrt(np.var(y) / H_var * sig0_initiate))
	sigma_1 = np.float32(math.sqrt(1/np.random.gamma(a_sigma,b_sigma)))
	sigma_e = np.float32(math.sqrt(1/np.random.gamma(a_e,b_e)))
	pie = np.float32(np.random.beta(pie_a,pie_b))

	if verbose > 0:
		print("There are %i k-mers in the model, and to set the background variation %f of the total phenotypic variation.\n We set the sigma 0 to be %f" %(H_c,sig0_initiate,sigma_0) )

		print("initiation for chain %i:" %(num) ,sigma_1,sigma_e,pie)
		
	it = 0
	burn_in_iter = 2000

	convergence_start_iter = burn_in_iter
	convergence_end_iter = burn_in_iter + 10000

	trace = np.empty((convergence_end_iter-burn_in_iter,6))
	top5_beta_trace = np.empty((convergence_end_iter-burn_in_iter,5))

	alpha = np.random.random(size = C_c).astype(np.float32)
	gamma = np.random.binomial(1,pie,H_c).astype(np.uint8)
	beta = np.array(np.zeros(H_c)).astype(np.float32)

	for i in range(H_c):
		if gamma[i] == 0:
			beta[i] = np.random.normal(0,sigma_0)
		else:
			beta[i] = np.random.normal(0,sigma_1) 

	#start sampling

	H_beta = np.matmul(H,beta).astype(np.float32)
	C_alpha = np.matmul(C,alpha).astype(np.float32)

	## precompute some variables 

	C_norm_2 = np.sum(C**2,axis=0).astype(np.float32)
	H_norm_2 = uf.col_norm2_chunked(H, chunk_rows=2000).astype(np.float32)
	#print(gamma.dtype,beta.dtype,H.dtype,C.dtype,H_norm_2.dtype,C_norm_2.dtype,sigma_0.dtype,sigma_1.dtype,sigma_e.dtype)

	convergence_container[num] = 0

	np.seterr(over='raise', invalid='raise')

	while it < convergence_end_iter:
		before = time.time()

		sigma_1 = sample_sigma_1(beta,gamma,a_sigma,b_sigma)
		pie = sample_pie(gamma,pie_a,pie_b)
		sigma_e = sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
		#gamma = sample_gamma(beta,sigma_0,sigma_1,pie)
		gamma = sample_gamma_numba(beta, sigma_0, sigma_1, pie, gamma)
		alpha,C_alpha = sample_alpha(y,H_beta,C_alpha,C,alpha,sigma_e,C_norm_2)
		beta,H_beta = sample_beta_numba_optimized(y,C_alpha,H_beta,H,beta,gamma,sigma_0,sigma_1,sigma_e,H_norm_2)
		genetic_var = np.var(H_beta)
		pheno_var = np.var(y - C_alpha)
		#large_beta = np.absolute(beta) > 0.1
		#large_beta_ratio = np.sum(large_beta) / len(beta)
		beta_p99 = np.percentile(np.absolute(beta), 99)
		total_heritability = genetic_var / pheno_var
		alpha_norm = np.linalg.norm(alpha, ord=2)
		beta_norm = np.linalg.norm(beta, ord=2)

		after = time.time()
		if (it > burn_in_iter and total_heritability > 1):
			if verbose > 0:
				print("unrealistic beta sample",it,genetic_var,pheno_var,total_heritability)
			continue

		else:
			if verbose > 1:
				print(num,it,str(after - before),sigma_1,sigma_e,beta_p99,total_heritability,np.sum(gamma))

			if it >= burn_in_iter:
				trace[it-burn_in_iter,:] = [sigma_e,beta_p99,alpha_norm,beta_norm,total_heritability,np.sum(gamma)]
				top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

			if it == convergence_end_iter - 1:
				convergence_scores = uf.convergence_geweke_test(trace,top5_beta_trace,convergence_start_iter-burn_in_iter,convergence_end_iter-burn_in_iter)

				if convergence_scores == 1:
					convergence_container[num] = 1
					if verbose > 0:
						print("convergence has been reached at %i iterations for chain %i. The MCMC Chain has entered a stationary stage" %(it,num))
						print("trace values:", trace[it-burn_in_iter,:])
					break
				else:
					trace_ = np.empty((1000,6))
					top5_beta_trace_ = np.empty((1000,5))


					trace = np.concatenate((trace[-(convergence_end_iter - burn_in_iter-1000):,:],trace_),axis=0)
					top5_beta_trace = np.concatenate((top5_beta_trace[-(convergence_end_iter - burn_in_iter-1000):,:],top5_beta_trace_),axis = 0)

					burn_in_iter += 1000
					convergence_start_iter += 1000
					convergence_end_iter += 1000

					#print(it,burn_in_iter,convergence_iter,convergence_start_iter,convergence_end_iter,trace.shape)

			it += 1

			if it > 100000: 
				convergence_container[num] = 0
				break

	if convergence_container[num] == 1:

		## MCMC draws for posterior mean

		posterior_draws = 10000

		alpha_mean = np.zeros(C_c)
		beta_mean = np.zeros(H_c)
		gamma_sum = np.zeros(H_c)

		alpha_M2 = np.zeros(C_c)
		beta_M2 = np.zeros(H_c)

		posterior_trace = np.empty((posterior_draws,6))

		alpha_trace = np.empty((posterior_draws,C_c))

		it = 0

		while it < posterior_draws:
		
			before = time.time()
			sigma_1 = sample_sigma_1(beta,gamma,a_sigma,b_sigma)
			pie = sample_pie(gamma,pie_a,pie_b)
			sigma_e = sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
			gamma = sample_gamma_numba(beta, sigma_0, sigma_1, pie, gamma)
			alpha,C_alpha = sample_alpha(y,H_beta,C_alpha,C,alpha,sigma_e,C_norm_2)
			beta,H_beta = sample_beta_numba_optimized(y,C_alpha,H_beta,H,beta,gamma,sigma_0,sigma_1,sigma_e,H_norm_2)
			genetic_var = np.var(H_beta)
			pheno_var = np.var(y - C_alpha)
			#large_beta = np.absolute(beta) > 0.1
			#large_beta_ratio = np.sum(large_beta) / len(beta)
			beta_p99 = np.percentile(np.absolute(beta), 99)
			total_heritability = genetic_var / pheno_var
			alpha_norm = np.linalg.norm(alpha, ord=2)
			beta_norm = np.linalg.norm(beta, ord=2)
			after = time.time()
			if total_heritability > 1:
				if verbose > 0:
					print("unrealistic beta sample",it,genetic_var,pheno_var,total_heritability)
				continue

			else:
				if verbose >1 :
					print(it,str(after - before),pie,sigma_1,sigma_e,np.sum(gamma),beta_p99,max(abs(beta)),total_heritability)
				posterior_trace[it,:] = [sigma_e,beta_p99,alpha_norm,beta_norm,total_heritability,np.sum(gamma)]
				alpha_trace[it,:] = alpha
				beta_mean,beta_M2 = uf.welford(beta_mean,beta_M2,beta,it)
				alpha_mean,alpha_M2 = uf.welford(alpha_mean,alpha_M2,alpha,it)
				gamma_sum += gamma

				if verbose > 0:
					if it > 0 and it % 2000 == 0:
						print("Posterior draws: %i iterations have been sampled for chain %i" %(it,num), str(after - before),posterior_trace[it,:])
				it += 1

		trace_container[num] = posterior_trace

		#alpha values
		alpha_container[num] = {'avg': alpha_mean,
								'M2': alpha_M2}

		#beta values
		beta_container[num] = {'avg':beta_mean,
								'M2':beta_M2}

		gamma_container[num] = gamma_sum / posterior_draws

	else:
		trace_container[num] = []

		#alpha values
		alpha_container[num] = {'avg': [],
								'M2': []}

		#beta values
		beta_container[num] = {'avg':[],
								'M2':[]}

		gamma_container[num] = []

