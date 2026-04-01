import re
import argparse
import numpy as np
import sys
import pandas as pd
import os
import array
from Bio.Seq import Seq
from sklearn import metrics
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as sch
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
import time
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix, save_npz
import pydustmasker

import geweke

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
	parser.add_argument('-b',type = float, action = 'store', dest = 'pi_b',default = 0.1, help = "pi_b for the beta distribution")
	parser.add_argument('-n',type = int, action = 'store', default = 8, dest = "num",help = "the number of threads for kmer counting / MCMC chains. Recommend at least 5")
	parser.add_argument('-v',type = int, action = 'store', default = 0, dest = 'verbose', help = "verbose levels 0: no stdout; 1: convergence and minimal stdout; 2: per MCMC iteration stdout")
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

# def generate_DM_sparse(sequences,sorted_kmers,k,n):
# 	start_time = time.time()

# 	r = len(sequences)
# 	c = len(sorted_kmers)
# 	sequence_names = list(sequences.keys())

# 	# Create a lookup dictionary for kmer positions
# 	kmer_to_index = {kmer: idx for idx, kmer in enumerate(sorted_kmers)}
# 	print(f"Processing {r} sequences for {c} kmers...")

# 	rows = []
# 	cols = []
# 	data = []

# 	for i, seq_name in enumerate(sequence_names):
# 		if i > 0 and i % 1000 == 0:
# 					print(f"Processed {i}/{r} sequences...")
		
# 		sequence = sequences[seq_name]
# 		kmer_counts = defaultdict(int)

# 		start = 0
# 		end = start + k

# 		while( start < len(sequence) - k + 1):
# 			current_kmer = sequence[start:end]
# 			idx = kmer_to_index.get(current_kmer)
# 			if idx is None:
# 				sys.exit("ERROR: FOUND A KMER that does not exist in the sequence")
# 			kmer_counts[idx] += 1

# 			start = start + 1 + n
# 			end = start + k
# 		for idx, count in kmer_counts.items():
# 			rows.append(i)
# 			cols.append(idx)
# 			data.append(count)
	
# 	DM_matrix = coo_matrix((data, (rows, cols)), shape=(r, c), dtype=np.uint16).tocsr()
# 	elapsed_time = time.time() - start_time
# 	print(f"Finished counting unique kmer dosage for all sequences in {elapsed_time:.2f} seconds.")
# 	return(sequence_names,DM_matrix)

def generate_DM_sparse_optimized(sequences,sorted_kmers,k,n):
	start_time = time.time()

	r = len(sequences)
	c = len(sorted_kmers)
	sequence_names = list(sequences.keys())

	# Create a lookup dictionary for kmer positions
	kmer_to_index = {kmer: idx for idx, kmer in enumerate(sorted_kmers)}
	print(f"Processing {r} sequences for {c} kmers...")

	# --- MEMORY OPTIMIZATION ---
	# 'H' = uint16 (Max count of 65,535 per kmer)
	# 'i' = int32  (Max 2.14 billion unique kmers/columns)
	# 'q' = int64  (Prevents matrix collapse when total non-zero elements > 2.14 billion)
	data = array.array('H')     
	indices = array.array('i')
	indptr = array.array('q', [0]) # Always starts at 0

	for i, seq_name in enumerate(sequence_names):
		if i > 0 and i % 1000 == 0:
					print(f"Processed {i}/{r} sequences...")
		
		sequence = sequences[seq_name]
		kmer_counts = defaultdict(int)

		for s in range(0, len(sequence) - k + 1, n + 1):
			kmer = sequence[s:s+k]
			if kmer in kmer_to_index:
				kmer_counts[kmer_to_index[kmer]] += 1
			else:
				sys.exit("ERROR: FOUND A KMER that does not exist in the sequence")

		for idx, count in kmer_counts.items():
			indices.append(idx)
			data.append(count)
		
		indptr.append(len(indices))

	np_data = np.frombuffer(data, dtype=np.uint16)
	np_indices = np.frombuffer(indices, dtype=np.int32)
	np_indptr = np.frombuffer(indptr, dtype=np.int64)

	del data, indices, indptr
	
	DM_matrix = csr_matrix((np_data, np_indices, np_indptr), shape=(r, c))
	elapsed_time = time.time() - start_time
	print(f"Finished counting unique kmer dosage for all sequences in {elapsed_time:.2f} seconds.")
	return(sequence_names,DM_matrix)


# def generation_cluster_DM(dosage,output):

# 	start_time = time.time()

# 	file = output + "_kmer_clusters.clstr"
# 	cluster = {}

# 	with open(file,"r") as FILE:
# 		for line in FILE:

# 			line = line.strip("\n")

# 			## search for > for the header

# 			if line.startswith(">"):
# 				name = line[1:]
# 				name = name.replace(" ", "_")

# 				if name not in cluster:
# 					cluster[name] = []
# 				else:
# 					sys.exit("ERROR: There are duplicated cluster names. Please double check! ")

# 			else:
# 				if name is None:
# 					sys.exit("ERROR: The cd-hit cluster output format is incorrect.")

# 				match = re.search(r"kmer_(\d+)",line)
# 				if match:
# 					cluster[name].append(int(match.group(1)))
# 				else:
# 					sys.exit("ERROR: incorrect regex.")

# 	r,c = dosage.shape
# 	cluster_count = len(cluster)
# 	cluster_names = list(cluster.keys())

# 	# Create a binary cluster mapping matrix
# 	#cluster_map = np.zeros((c, cluster_count), dtype=int)
# 	cluster_map = lil_matrix((c, cluster_count), dtype=np.int8)

# 	for idx, key in enumerate(cluster_names):
# 		cluster_indices = np.array(cluster[key])
# 		cluster_map[cluster_indices, idx] = 1  # Mark k-mers in each cluster

# 	# cluster_dosage = dosage @ cluster_map

# 	# elapsed_time = time.time() - start_time
# 	# print(f"Finished calculating kmer cluster dosage matrix in {elapsed_time:.2f} seconds.")

# 	#dosage_sparse = csc_matrix(dosage,dtype=np.int32)
# 	#cluster_map_sparse = csr_matrix(cluster_map,dtype=np.int32)
# 	cluster_map_sparse = cluster_map.tocsr()


# 	#cluster_dosage_1 = dosage_sparse.dot(cluster_map_sparse)
# 	cluster_dosage_1 = dosage.dot(cluster_map_sparse)
# 	cluster_dosage_1_np = np.matrix(cluster_dosage_1.toarray())
# 	elapsed_time = time.time() - start_time

# 	print(f"Finished calculating kmer cluster dosage matrix in {elapsed_time:.2f} seconds.")

# 	return(cluster_dosage_1_np,cluster_names)

def generation_cluster_DM_optimized(dosage,output):

	start_time = time.time()

	file = output + "_kmer_clusters.clstr"
	cluster_names = []
	seen_names = set()

	rows = array.array('i')
	cols = array.array('i')
	current_cluster_idx = -1

	with open(file,"r") as FILE:
		for line in FILE:

			line = line.strip("\n")

			## search for > for the header

			if line.startswith(">"):
				name = line[1:]
				name = name.replace(" ", "_")

				if name in seen_names:
					sys.exit("ERROR: There are duplicated cluster names. Please double check! ")
				seen_names.add(name)
				cluster_names.append(name)
				current_cluster_idx += 1
					

			else:
				match = re.search(r"kmer_(\d+)",line)

				if match:
					rows.append(int(match.group(1)))
					cols.append(current_cluster_idx)
				else:
					sys.exit("ERROR: incorrect regex.")

	r,c = dosage.shape
	cluster_count = len(cluster_names)

	np_rows = np.frombuffer(rows, dtype=np.int32)
	np_cols = np.frombuffer(cols, dtype=np.int32)
	np_data = np.ones(len(rows), dtype=np.int8)

	del rows, cols, seen_names
	cluster_map_sparse = coo_matrix((np_data, (np_rows, np_cols)), shape=(c, cluster_count)).tocsc()
	
	cluster_dosage = dosage.dot(cluster_map_sparse)
	elapsed_time = time.time() - start_time

	print(f"Finished calculating kmer cluster dosage matrix in {elapsed_time:.2f} seconds.")

	return(cluster_dosage,cluster_names)

def read_input_files(geno, pheno,covar):

	X = pd.read_csv(str(geno),sep=",")
	n,p = X.shape

	kmer_names = np.array(X.columns.values.tolist())

	X_nparray = np.array(X,dtype=np.uint32)

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
	return(y,X_nparray,kmer_names,C,covariate_names)

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

def convergence_geweke_test(trace,top5_beta_trace,start,end):
    max_z = []

    ## convergence for the trace values
    n = trace.shape[1]
    for t in range(n):
        trace_convergence = trace[start:end,t]
        trace_t_convergence_zscores = geweke.geweke(trace_convergence)[:,1]
        max_z.append(np.amax(np.absolute(trace_t_convergence_zscores)))

    m = top5_beta_trace.shape[1]
    for b in range(m):
        top_beta_convergence = top5_beta_trace[start:end,b]
        beta_b_convergence_zscores = geweke.geweke(top_beta_convergence)[:,1]
        max_z.append(np.amax(np.absolute(beta_b_convergence_zscores)))

    if np.amax(max_z) < 1.5:
        return(1)

def welford(mean,M2,x,n):
    n = n + 1

    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    M2 += delta * delta2

    return(mean,M2)

def merge_welford(A_mean, A_M2,A_n,B_mean,B_M2,B_n):

    if A_n == 0:
        return(B_mean,B_M2,B_n)

    if B_n == 0:
        return(A_mean,A_M2,A_n)
    
    n_new = A_n + B_n
    delta = B_mean - A_mean

    mean = A_mean + delta * (B_n / n_new)
    M2 = A_M2 + B_M2 + (delta * delta) * (A_n * B_n / n_new)

    return(mean,M2,n_new)



def col_norm2_chunked(H, chunk_rows=2000, out_dtype=np.float64):
	n, p = H.shape
	out = np.zeros(p, dtype=out_dtype)

	for s in range(0, n, chunk_rows):
		block = H[s:s+chunk_rows, :].astype(np.float32, copy=False)
		# sum of squares per column for this block
		out += np.einsum('ij,ij->j', block, block)

	return(out)

def compute_sequence_context_doublestrand(sequences):

	low_complexity_fraction = {}
	GC_content = {}
	CG_density = {}
	CHH_density = {}
	CHG_density = {}
	length = {}
	CG_obs_exp = {}
	CHH_obs_exp = {}
	CHG_obs_exp = {}


	for name in sequences:
		seq = sequences[name].upper()
		rc = str(Seq(seq).reverse_complement())
		L = len(seq)
		length[name] = L

		# base counts (forward strand)
		C = seq.count("C")
		G = seq.count("G")
		A = seq.count("A")
		T = seq.count("T")
		H_f = A + C + T

		# reverse-strand mono-nucleotide counts
		C_r, G_r = G, C
		H_r = T + G + A

		CG_f, CHG_f, CHH_f = count_contexts_per_seq(seq)
		CG_r, CHG_r, CHH_r = count_contexts_per_seq(rc)

		# strand-invariant totals
		CG_total = CG_f + CG_r
		CHG_total = CHG_f + CHG_r
		CHH_total = CHH_f + CHH_r
		
		GC_content[name] = (C + G) / L
		CG_density[name] = CG_total / (2 * (L - 1)) 
		CHG_density[name] = CHG_total / (2 * (L - 2)) 
		CHH_density[name] = CHH_total / (2 * (L - 2))
		
		## low complexity fraction
		masker = pydustmasker.DustMasker(seq)
		low_complexity_fraction[name] = masker.n_masked_bases / L


		## expected vs observed for CG, CHG and CHH

		## CG
		expected_CG = (C * G) * (L-1) / L / L
		CG_obs_exp[name] = CG_f / expected_CG if expected_CG > 0 else 0


		## CHG
		expected_CHG_f = (C * H_f * G * (L - 2)) / (L ** 3)
		expected_CHG_r = (C_r * H_r * G_r * (L - 2)) / (L ** 3)
		expected_CHG_total = expected_CHG_f + expected_CHG_r
		CHG_obs_exp[name] = CHG_total / expected_CHG_total if expected_CHG_total > 0 else 0

		## CHH
		expected_CHH_f = (C * (H_f ** 2) * (L - 2)) / (L ** 3)
		expected_CHH_r = (C_r * (H_r ** 2) * (L - 2)) / (L ** 3)
		expected_CHH_total = expected_CHH_f + expected_CHH_r
		CHH_obs_exp[name] = CHH_total / expected_CHH_total if expected_CHH_total > 0 else 0

	return(GC_content, CG_density, CHG_density, CHH_density, length,low_complexity_fraction,CG_obs_exp,CHG_obs_exp,CHH_obs_exp)

_RE_CHG = re.compile(r'(?=(C[ACT]G))')
_RE_CHH = re.compile(r'(?=(C[ACT][ACT]))')

def count_contexts_per_seq(sequence):
	SEQ = sequence.upper() ## all upper case

	CG  = SEQ.count("CG")                     # fast C routine
	CHG = len(_RE_CHG.findall(SEQ))           # overlapping matches
	CHH = len(_RE_CHH.findall(SEQ))

	return(CG,CHG,CHH)
















