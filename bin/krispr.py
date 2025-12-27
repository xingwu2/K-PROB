
import os
import numpy as np
import pandas as pd
import sys
import time

#import utility scripts

import utility_functions as uf
import kmer_clustering as cluster
import multiprocessing as mp
import spike_point_mass as sp_pointmass
import spike_normal as sp_normal

def main():

	DIR = os.path.realpath(os.path.dirname(__file__))

	args = uf.parse_arguments()

	if args.task == "count":

		## STEP 1: Read multi-fasta file and store them in a dict

		sequences = uf.read_fasta_file(args.sequence)

		## STEP 2: identify unique kmers from the sequences and sort Kmers based on frequency and lexicography

		kmer_counts = uf.count_kmers_from_seq(sequences,args.k,args.gap)
		print("Finished counting unique k-mers. Identified %d unique kmers in total" %(len(kmer_counts)))

		sorted_kmers = sorted(kmer_counts.keys(), key=lambda km: (-kmer_counts[km], km))
		
		OUTPUT_UNIQUE_KMERS = open(args.output+"_unique_kmers.fa","w")
		for i, kmer in enumerate(sorted_kmers):
			print("%s\n%s" %(">kmer_"+str(i),sorted_kmers[i]),file = OUTPUT_UNIQUE_KMERS)
		OUTPUT_UNIQUE_KMERS.close()

		## STEP 3: perform sequence based clustering on unique kmers (default = True)

		similarity = 1-(1.0/float(args.k)) - 0.01

		kmer_clusters = cluster.cdhit_cluster(args.output+"_unique_kmers.fa",args.output+"_kmer_clusters",similarity,args.wordsize)


		## STEP 4: generate the kmer design matrix for each sequence, the number indicates the dosage of kmer and kmer_clusters

		sequence_names,dosage = uf.generate_DM(sequences,sorted_kmers,args.k,args.gap)

		cluster_dosage,cluster_names = uf.generation_cluster_DM(dosage,args.output)


		## identify the kmer clusters with mimimum frequency of occurrence (default 0.01)
		cluster_indicator = cluster_dosage > 0
		occurance_frequency = np.sum(np.array(cluster_indicator),axis=0) / cluster_dosage.shape[0]
		col_indices = np.where(occurance_frequency > args.cutoff)[0]
		cluster_dosage_passed = cluster_dosage[:,col_indices]
		cluster_names_passed = [cluster_names[i] for i in col_indices]

		cluster_dosage_passed_pd = pd.DataFrame(cluster_dosage_passed)
		cluster_dosage_passed_pd.to_csv(args.output+"_Cluster_DosageMatrix_occurrence_"+str(args.cutoff)+".csv",header=cluster_names_passed,index=False)
		
		if args.unique == True:
			dosage_pd = pd.DataFrame(dosage)
			#dosage_pd.index = sequence_names
			dosage_pd.to_csv(args.output+"_DosageMatrix.csv",header=sorted_kmers,index=False)


	elif args.task == "mapping":

		## The following script will perform the mapping algorithm to identify the causal 

		y, X, kmer_names,C, covariate_names = uf.read_input_files(args.geno,args.pheno,args.covar)

		trace_container = mp.Manager().dict()
		gamma_container = mp.Manager().dict()
		beta_container = mp.Manager().dict()
		alpha_container = mp.Manager().dict()

		processes = []

		if args.model == 1:
			for num in range(args.num):
				p = mp.Process(target = sp_normal.sampling,args=(args.verbose,y,C,X,args.s0,12000,args.output,num,trace_container,gamma_container,beta_container,alpha_container,args.pi_b))
				processes.append(p)
				p.start()

		else:
			for num in range(args.num):
				p = mp.Process(target = sp_pointmass.sampling, args=(args.verbose,y,C,X,12000,args.output,num,trace_container,gamma_container,beta_container,alpha_container,args.pi_b))
				processes.append(p)
				p.start()

		for process in processes:
			process.join()

		alpha_posterior = []
		alpha_posterior_sd = []
		beta_posterior = []
		beta_posterior_sd = []
		kmer_pip = []
		trace_posterior = []
		trace_posterior_sd = []

		for num in range(args.num):
			alpha_posterior.append(alpha_container[num]["avg"])
			alpha_posterior_sd.append(alpha_container[num]["sd"])
			beta_posterior.append(beta_container[num]["avg"])
			beta_posterior_sd.append(beta_container[num]["sd"])
			trace_posterior.append(trace_container[num]["avg"])
			trace_posterior_sd.append(trace_container[num]["sd"])
			kmer_pip.append(gamma_container[num]["kmer"])


		alpha_posterior_median = np.median(alpha_posterior,axis=0)
		alpha_posterior_sd_median = np.median(alpha_posterior_sd,axis=0)
		beta_posterior_median = np.median(beta_posterior,axis=0)
		beta_posterior_sd_median = np.median(beta_posterior_sd,axis=0)
		trace_posterior_median = np.median(trace_posterior,axis=0)
		trace_posterior_sd_median = np.median(trace_posterior_sd,axis=0)
		kmer_pip_median = np.median(kmer_pip,axis=0)


		## calculate FDR for different kmers
		index,kmer_fdr = uf.fdr_calculation(kmer_pip_median)
		
		## sort pip, kmer names, beta and beta_sd based on pip
		sorted_kmer_names = kmer_names[index]
		sorted_kmer_pip = kmer_pip_median[index]
		sorted_beta_median = beta_posterior_median[index]
		sorted_beta_sd_median = beta_posterior_sd_median[index]


		OUTPUT_KMER = open(args.output+"_kmer_pip.txt","w")
		print("%s\t%s\t%s" %("kmer_name","pip","fdr"),file = OUTPUT_KMER)
		for i in range(len(sorted_kmer_names)):
			print("%s\t%s\t%s" %(sorted_kmer_names[i],sorted_kmer_pip[i],kmer_fdr[i]),file = OUTPUT_KMER)
		OUTPUT_KMER.close()

		OUTPUT_ALPHA = open(args.output+"_alpha.txt","w")
		print("%s\t%s\t%s" %("covariate_name","covariate_effect","covariate_effect_sd"),file = OUTPUT_ALPHA)
		for i in range(len(alpha_posterior_median)):
			print("%s\t%f\t%f" %(covariate_names[i],alpha_posterior_median[i],alpha_posterior_sd_median[i]),file = OUTPUT_ALPHA)
		OUTPUT_ALPHA.close()


		OUTPUT_BETA = open(args.output+"_beta.txt","w")
		print("%s\t%s\t%s" %("kmer_name","kmer_effect","kmer_effect_sd"),file = OUTPUT_BETA)
		for i in range(len(sorted_kmer_names)):
			print("%s\t%f\t%f" %(sorted_kmer_names[i],sorted_beta_median[i],sorted_beta_sd_median[i]),file = OUTPUT_BETA)
		OUTPUT_BETA.close()


		trace_variables = ["Sigma_1","Sigma_e","Large_Effect_Kmer_Proportion","Variance_Explained","Num_Causal_Kmer"]
		OUTPUT_TRACE = open(args.output+"_trace.txt","w")
		for i in range(len(trace_posterior_median)):
			print("%s\t%f\t%f" %(trace_variables[i],trace_posterior_median[i],trace_posterior_sd_median[i]),file = OUTPUT_TRACE)
		OUTPUT_TRACE.close()

	else:
		sys.exit("ERROR: Please provide the name of the task: count or mapping. Details see the manual (-h).")



## run program
if __name__ == "__main__":
    main()
