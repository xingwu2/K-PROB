import re
import argparse
import numpy as np
import sys
import pandas as pd
import os
from Bio.Seq import Seq
from sklearn import metrics
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as sch
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
import time
from collections import defaultdict
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix




def parse_arguments():
	"""
    Parse command line arguments.
    """
	parser = argparse.ArgumentParser()
	parser.add_argument('-f',type = str, action= 'store',dest='sequence',help='the multi-fasta file')
	parser.add_argument('-k',type = int, action= 'store',dest='k',default=7,help = "size of the kmer")
	parser.add_argument('-g',type = int, action = 'store', dest = 'gap',default=0,help = "the number of nucleotide gap between 2 kmers")
	parser.add_argument('-l',type = float, action = 'store', dest = "cutoff",default = 0.01,help = "a separate DM matrix with the minimum occurrence frequency for a kmer/kmer cluster to be included in the  mapping (default:0.01)")


	parser.add_argument('-t',type = str, action = 'store', dest = 'task',help = "count | mapping")
	parser.add_argument('-u',action = 'store_true', dest = 'unique',default = False, help = "output the unique kmer dosage matrix (default: false)")


	parser.add_argument('-w',type = int, action = 'store', dest = 'wordsize',default=5,help = "the wordsize for cd-hit-est")
	parser.add_argument('-s',type = float, action = 'store', dest = 'similarity',default=0.9,help = "the similarity cutoff for cd-hit-est")

	parser.add_argument('-x',type = str, action = 'store', dest = 'geno',help = "the input matrix (X) for the mapping step")
	parser.add_argument('-c',type = str, action = 'store', dest = 'covar',help = "the covariates (C) for the mapping step")
	parser.add_argument('-y',type = str, action = 'store', dest = 'pheno',help = "the response variable for the mapping step")

	parser.add_argument('-m',type = int, action = 'store', dest = 'model',default = 1, help = "the statistical model for kmer effect estimation. Krispr offers two spike priors 1 (default): small effect around 0; 2: point mass at 0")
	parser.add_argument('-s0',type = float, action = 'store', dest = 's0',default = 0.1, help = "the proportion of phenotypic variation explained by background kmers")
	parser.add_argument('-b',type = float, action = 'store', dest = 'pi_b',default = 0.01, help = "pi_b for the beta distribution")
	parser.add_argument('-n',type = int, action = 'store', default = 8, dest = "num",help = "the number of threads for kmer counting / MCMC chains. Recommend at least 5")
	parser.add_argument('-v',action = 'store_true', dest = 'verbose',default = False, help = "print out each MCMC iteration")
	parser.add_argument('-o',type = str, action = 'store', dest = 'output',help = "the prefix of the output files")
	args = parser.parse_args()

	return(args)

def read_fasta_file(file):

	sequences = {}

	with open(file,"r") as FILE:
		for line in FILE:

			line = line.strip("\n")

			## search for > for the header

			if line.startswith(">"):
				name = line[1:]
				if name not in sequences:
					sequences[name] = ""
				else:
					print("There are multiple %s sequences." %(name))
					sys.exit("ERROR: There are duplicated names in your fasta file. Please double check! ")

			else:
				if name is None:
					sys.exit("ERROR: The fasta file format is incorrect. No header line found before sequence.")
				sequences[name] += line
	print("Finished loading sequences.")
	return(sequences)

def count_kmers_from_seq(sequences,k,n):

	kmer_counts = {}
	it = 1

	for key in sequences:

		sequence = sequences[key]
		
		l = len(sequence)

		start = 0

		end = start + k

		while( start < l - k + 1):

			kmer = sequence[start:end]
			#rev_comp = str(Seq(kmer).reverse_complement())
			#canonical_kmer = min(kmer, rev_comp)
			### identify unique kmers 
			if kmer not in kmer_counts:
				kmer_counts[kmer] = 1
			else:
				kmer_counts[kmer] += 1

			start = start + 1 + n
			end = start + k
		it += 1

		# if it %100 ==0 :
		# 	print("Processed %i sequences, found %i unique kmers so far" %(it,len(kmer_counts)))

	return(kmer_counts)

def kmer_one_hot_encoding(kmer):
	kmer_numeric = np.zeros(len(kmer)*4)

	for i in range(len(kmer)):
		if kmer[i] == "A" or kmer[i] == "a":
			kmer_numeric[i*4] = 1
		elif kmer[i] == "T" or kmer[i] == "t":
			kmer_numeric[i*4 + 1] = 1
		elif kmer[i] == "G" or kmer[i] == "g":
			kmer_numeric[i*4 + 2] = 1
		elif kmer[i] == "C" or kmer[i] == "c":
			kmer_numeric[i*4 + 3] = 1
		else:
			sys.exit("ERROR: found non-conventional nucleotide")
	return(kmer_numeric)

def kmer_one_hot_decoding(kmer_numeric):
	kmer = []

	k = int(len(kmer_numeric) / 4)

	for i in range(k):
		if kmer_numeric[i*4] == 1:
			kmer.append("A")
		elif kmer_numeric[i*4 + 1] == 1:
			kmer.append("T")
		elif kmer_numeric[i*4 + 2] == 1:
			kmer.append("G")
		elif kmer_numeric[i*4 + 3] == 1:
			kmer.append("C")
	kmer_decode = "".join(kmer)

	return(kmer_decode)

# def kmer_clustering(ALL_K_mers):

# 	k = len(ALL_K_mers[0])
# 	nrow = len(ALL_K_mers)
# 	ncol = k*4
# 	kmer_matrix = np.empty((nrow,ncol))

# 	one_mismatch = []

# 	for I in range(len(ALL_K_mers)):
# 		kmer = ALL_K_mers[I]
# 		kmer_numeric = kmer_one_hot_encoding(kmer)
# 		kmer_matrix[I,:] = kmer_numeric

# 	## Use manhattan distance to find reference kmers

# 	manhattan_distances = np.array(squareform(pdist(kmer_matrix, metric='cityblock')),dtype=int)
# 	one_mismatch = np.sum(manhattan_distances == 2,axis=1)

# 	kmer_rank_index = np.array(np.argsort(one_mismatch)[::-1],dtype=int)

# 	K_mer_clusters = {}

# 	ALL_K_mers_tmp = [ALL_K_mers[i] for i in kmer_rank_index]

# 	index_to_be_remove_1 = []

# 	## with 1 mismatch
# 	for I in range(len(kmer_rank_index)):

# 		index = kmer_rank_index[I]

# 		kmer_ = ALL_K_mers[index]

# 		if kmer_ in  ALL_K_mers_tmp:

# 			#kmers with 1 mismatch to I only
# 			index1 = np.where(manhattan_distances[index,:] ==2 )[0]
# 			# initiate the cluster with a reference
# 			K_mer_clusters[kmer_] = [kmer_]
# 			for J in index1:
# 				if sum(manhattan_distances[J,:] ==2) == 1:
# 					K_mer_clusters[kmer_].append(ALL_K_mers[J])
# 					ALL_K_mers_tmp.remove(ALL_K_mers[J])
# 					index_to_be_remove_1.append(J)

# 	kmer_rank_index = [item for item in kmer_rank_index if item not in index_to_be_remove_1]
# 	manhattan_distances_removed = np.delete(manhattan_distances,index_to_be_remove_1,axis=1)
# 	print(manhattan_distances_removed.shape)

# 	## with 2 mismatch

# 	for I in range(len(kmer_rank_index)):

# 		index = kmer_rank_index[I]

# 		kmer_ = ALL_K_mers[index]
# 		if kmer_ in  ALL_K_mers_tmp:

# 			#kmers with 1 mismatch to I only
# 			index1 = np.where(manhattan_distances[index,:] == 4 )[0]
# 			index1 = [item for item in index1 if item not in index_to_be_remove_1]
# 			# initiate the cluster with a reference
# 			#key_name = "Kmer_cluster_"+str(I)
# 			#K_mer_clusters[key_name] = [kmer_]
# 			for J in index1:
# 				#print(J,sum(manhattan_distances_removed[J,:] == 4))
# 				if sum(manhattan_distances_removed[J,:] == 4) == 1:
# 					print(J,kmer_,ALL_K_mers[J])
# 					K_mer_clusters[kmer_].append(ALL_K_mers[J])
# 					ALL_K_mers_tmp.remove(ALL_K_mers[J])
	
# 	print(len(K_mer_clusters))

# 	# Get the first 5 key-value pairs
# 	first_five = list(K_mer_clusters.items())[:5]

# 	# Print them
# 	for key, value in first_five:
# 		print(key, value)




	# for I in range(len(ALL_K_mers)):
		# one_mismatch.append(sum(manhattan_distances[I,:] == 2))

	#print(one_mismatch)


	
	# print(ALL_K_mers[0],ALL_K_mers[119])
	# linkage_matrix = sch.linkage(kmer_matrix, method='single')
	# print(linkage_matrix.shape)
	# clusters = sch.fcluster(linkage_matrix, t=1.5, criterion='distance')
	# print(len(clusters))

	# dists = squareform(pdist((kmer_matrix)))
	# affinity_matrix = np.exp( - dists * dists )
	# np.fill_diagonal(affinity_matrix, 0)

	#print(ALL_K_mers[:10],kmer_matrix[:10,:])

	##manhattan_distances = squareform(pdist(kmer_matrix, metric='cityblock'))
	#print(sum(manhattan_distances[0,:]==2))

	# connectivity = kneighbors_graph(kmer_matrix, n_neighbors=k*3, mode='connectivity', include_self=False)
	# connectivity_matrix = connectivity.toarray()
	# weighted_connectivity = np.multiply(affinity_matrix,connectivity_matrix)
	# #G = knn_graph.toarray()
	# d = np.sum(weighted_connectivity,axis=1)
	# print(np.unique(d))
	# print(np.sort(d)[::-1])
	# print(sum(d==max(d)))


def generate_DM(sequences,sorted_kmers,k,n):

	start_time = time.time()

	r = len(sequences)
	c = len(sorted_kmers)

	DM_matrix = np.zeros((r,c),dtype=np.uint16)

	# Create a lookup dictionary for kmer positions
	kmer_to_index = {kmer: idx for idx, kmer in enumerate(sorted_kmers)}
	#print(kmer_to_index)

	sequence_names = list(sequences.keys())

	print(f"Processing {r} sequences for {c} kmers...")

	# Process each sequence
	for i, seq_name in enumerate(sequence_names):
		if i > 0 and i % 100 == 0:
			print(f"Processed {i}/{r} sequences...")
            
		sequence = sequences[seq_name]

		# Count all kmers in one pass through the sequence
		kmer_counts = defaultdict(int)

		# Use sliding window to count all kmers in the sequence at once

		start = 0

		end = start + k

		while( start < len(sequence) - k + 1):
			current_kmer = sequence[start:end]
			#rev_comp = str(Seq(current_kmer).reverse_complement())
			#canonical_kmer = min(current_kmer, rev_comp)
			if current_kmer in kmer_to_index:  
				kmer_counts[current_kmer] += 1
			else:
				#print(current_kmer)
				sys.exit("ERROR: FOUND A KMER that does not exist in the sequence")

			start = start + 1 + n
			end = start + k

		# Fill the matrix with counts
		for kmer, count in kmer_counts.items():
			DM_matrix[i, kmer_to_index[kmer]] = count

	elapsed_time = time.time() - start_time
	
	print(f"Finished counting unique kmer dosage for all sequences in {elapsed_time:.2f} seconds.")

	return(sequence_names,DM_matrix)


def generation_cluster_DM(dosage,output):

	start_time = time.time()

	file = output + "_kmer_clusters.clstr"
	cluster = {}

	with open(file,"r") as FILE:
		for line in FILE:

			line = line.strip("\n")

			## search for > for the header

			if line.startswith(">"):
				name = line[1:]
				name = name.replace(" ", "_")

				if name not in cluster:
					cluster[name] = []
				else:
					sys.exit("ERROR: There are duplicated cluster names. Please double check! ")

			else:
				if name is None:
					sys.exit("ERROR: The cd-hit cluster output format is incorrect.")

				match = re.search(r"kmer_(\d+)",line)
				if match:
					cluster[name].append(int(match.group(1)))
				else:
					sys.exit("ERROR: incorrect regex.")

	r,c = dosage.shape
	cluster_count = len(cluster)
	cluster_names = list(cluster.keys())

	# Create a binary cluster mapping matrix
	#cluster_map = np.zeros((c, cluster_count), dtype=int)
	cluster_map = lil_matrix((c, cluster_count), dtype=np.int8)

	for idx, key in enumerate(cluster_names):
		cluster_indices = np.array(cluster[key])
		cluster_map[cluster_indices, idx] = 1  # Mark k-mers in each cluster

	# cluster_dosage = dosage @ cluster_map

	# elapsed_time = time.time() - start_time
	# print(f"Finished calculating kmer cluster dosage matrix in {elapsed_time:.2f} seconds.")

	dosage_sparse = csc_matrix(dosage,dtype=np.int32)
	#cluster_map_sparse = csr_matrix(cluster_map,dtype=np.int32)
	cluster_map_sparse = cluster_map.tocsr()


	cluster_dosage_1 = dosage_sparse.dot(cluster_map_sparse)
	cluster_dosage_1_np = np.matrix(cluster_dosage_1.toarray())
	elapsed_time = time.time() - start_time

	print(f"Finished calculating kmer cluster dosage matrix in {elapsed_time:.2f} seconds.")

	return(cluster_dosage_1_np,cluster_names)

def read_input_files(geno, pheno,covar):

	X = pd.read_csv(str(geno),sep=",")
	n,p = X.shape

	kmer_names = np.array(X.columns.values.tolist())

	y = []
	with open(str(pheno),"r") as f:
		for line in f:
			line = line.strip("\n")
			y.append(float(line))

	y = np.asarray(y)

	if covar is None:
		C = np.ones(n)
		C = C.reshape(n, 1)
		covariate_names = ["intercept"]

	else:
		C =  pd.read_csv(str(covar),sep=",")
		covariate_names =  np.array(C.columns.values.tolist())
		C = np.array(C)
	return(y,X,kmer_names,C,covariate_names)

def fdr_calculation(kmer_pip_median):

	ordered_index = np.argsort(kmer_pip_median)[::-1]

	sorted_kmer_pip = kmer_pip_median[ordered_index]

	## FDR calculation from sorted pip values

	fdr = []

	for i in range(len(sorted_kmer_pip)):
		if i == 0:
			fdr.append( 1 - sorted_kmer_pip[i])
		else:
			fdr.append( 1 - np.mean(sorted_kmer_pip[:i+1]))

	return(ordered_index,fdr)



		
















