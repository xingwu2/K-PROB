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
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix, save_npz, load_npz
import pydustmasker
import h5py

import geweke

def parse_arguments():
	"""
	Parse command line arguments using subparsers for distinct tasks.
	"""
	# 1. Create a base parser for global arguments to avoid repeating them
	# add_help=False is necessary here so the subcommands can handle help normally
	base_parser = argparse.ArgumentParser(add_help=False)
	base_parser.add_argument('-o', type=str, action='store', dest='output', required=True, 
							 help="the prefix of the output files")
	
	parser = argparse.ArgumentParser(description="Krispr Pipeline")

	# dest='task' acts exactly like your old -t argument. 
	# It stores the chosen subparser name (count, decompose, or mapping) into args.task
	subparsers = parser.add_subparsers(dest='task', required=True, help="Task to execute")

	# ---------------------------------------------------------
	# SUBCOMMAND 1: COUNT
	# ---------------------------------------------------------
	parser_count = subparsers.add_parser('count', parents=[base_parser], help="Run the count task")
	parser_count.add_argument('-f', type=str, action='store', dest='sequence', required=True, help='the multi-fasta file')
	parser_count.add_argument('-k', type=int, action='store', dest='k', required=True, help="size of the kmer")
	parser_count.add_argument('-g', type=int, action='store', dest='gap', default=0, help="nucleotide gap between 2 kmers. Default: 0")
	parser_count.add_argument('-l', type=float, action='store', dest="cutoff", default=0.01, help="minimum occurrence frequency. Default: 0.01")
	parser_count.add_argument('-u', action='store_true', dest='unique', default=False, help="output unique kmer dosage matrix. Default: false")
	parser_count.add_argument('-w', type=int, action='store', dest='wordsize', default=5, help="wordsize for cd-hit-est. Default: 5")
	parser_count.add_argument('-s', type=float, action='store', dest='similarity', default=1.0, help="number of mismatches allowed in the kmer for cd-ht-est to cluster unique kmers. Default: 1")

	# ---------------------------------------------------------
	# SUBCOMMAND 2: DECOMPOSE
	# ---------------------------------------------------------
	parser_decompose = subparsers.add_parser('decompose', parents=[base_parser], help="Run the decompose task")
	parser_decompose.add_argument('-d', type=str, action='store', dest='dm', required=True, help="clustered dosage matrix .npz file from the count task")
	parser_decompose.add_argument('-a', type=str, action='store', dest='allele', required=True, help="gene alleles.txt from the count task")
	parser_decompose.add_argument('-r', type=str, action='store', dest='kmer_cluster', required=True, help="kmer_cluster_{cutoff}.txt from the count task")
	parser_decompose.add_argument('-p', type=str, action='store', dest='promoter_feature', required=True, help="allele-level promoter feature table")
	parser_decompose.add_argument('-e', type=str, action='store', dest='expression', required=True, help="gene expression table")


	# ---------------------------------------------------------
	# SUBCOMMAND 3: MAPPING
	# ---------------------------------------------------------
	parser_mapping = subparsers.add_parser('mapping', parents=[base_parser], help="Run the mapping task")
	parser_mapping.add_argument('-x', type=str, action='store', dest='geno', required=True, help="input matrix (X)")
	parser_mapping.add_argument('-c', type=str, action='store', dest='covar', help="covariates (C)")
	parser_mapping.add_argument('-y', type=str, action='store', dest='pheno', required=True, help="gene expression values")
	parser_mapping.add_argument('-m', type=int, action='store', dest='model', default=1, help="statistical model (1) two normal mixture (2) point mass at 0. Default: 1)")
	parser_mapping.add_argument('-s0', type=float, action='store', dest='s0', default=0.1, help="proportion of phenotypic variation explained. Default: 0.1")
	parser_mapping.add_argument('-b', type=float, action='store', dest='pi_b', default=0.1, help="pi_b for the beta distribution. Default: 0.1")
	parser_mapping.add_argument('-n', type=int, action='store', dest='num', default=8, help="number of threads. Recommend at least 5. Default: 8)")
	parser_mapping.add_argument('-v', type=int, action='store', default=0, dest='verbose',help="verbose levels 0: no stdout; 1: minimal; 2: detailed. Default: 0)")
	args = parser.parse_args()
	return args

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

def expression_decompose_memory_optimized(dm, allele, kmer_cluster, expression, promoter_feature,output):
	## load expression data, genotype data and gene features
	mat = load_npz(dm)
	alleles = pd.read_csv(allele, header=None, names=["Allele"])
	kmer_clusters = pd.read_csv(kmer_cluster, header=None)[0].tolist()

	dosage_df = pd.DataFrame.sparse.from_spmatrix(mat, columns=kmer_clusters)
	dosage_df.index = alleles["Allele"]
	
	# Safely renamed to prevent overwriting the string arguments
	expr_df = pd.read_csv(expression, sep=",")
	promoter_feat_df = pd.read_csv(promoter_feature, sep=",")

	## first run some QC on the input data
	expression_column_names = expr_df.columns.to_list()
	if expression_column_names != ["Individual", "Gene", "Allele", "Expression"]:
		sys.exit("ERROR: The expression table should have the EXACT following columns and column names: Individual, Gene, Allele, Expression")
	if sum(promoter_feat_df["Allele"] != expr_df["Allele"]) > 0:
		sys.exit("ERROR: The gene-allele names in the promoter feature table do not match the gene-allele names in the expression table. Please double check! ")
	if dosage_df.index.to_list() != expr_df["Allele"].tolist():
		sys.exit("ERROR: The gene-allele names in the dosage matrix do not match the gene-allele names in the expression table. Please double check! ")
	
	## concat expression and promoter feature
	expression_promoter_df = pd.concat([expr_df, promoter_feat_df.drop(columns=["Allele"])], axis=1)

	print("Finished QC and all tables are in order. Starting expression decomposition...\n")

	## expression decompose y_ij = \mu + \alpha_i  + \gamma_j + \delta_ij 
	expression_mu = expression_promoter_df["Expression"].mean()
	expression_promoter_df['alpha_i'] = expression_promoter_df.groupby('Gene',sort = False)['Expression'].transform('mean') - expression_mu
	expression_promoter_df['gamma_j'] = expression_promoter_df.groupby('Individual',sort = False)['Expression'].transform('mean') - expression_mu
	expression_promoter_df['delta_ij'] = expression_promoter_df['Expression'] - expression_mu - expression_promoter_df['alpha_i'] - expression_promoter_df['gamma_j']
	
	expression_promoter_df['per_genes_sd'] = expression_promoter_df.groupby("Gene",sort = False)["delta_ij"].transform("std")
	global_sd = expression_promoter_df.groupby("Gene",sort = False)['per_genes_sd'].first().median()
	safe_sd = expression_promoter_df["per_genes_sd"].fillna(global_sd).clip(lower=global_sd*0.1)
	expression_promoter_df['delta_ij_scaled'] = expression_promoter_df['delta_ij'] / safe_sd

	print("Finished calculating the allelic deviation. Starting chunked double-centering to HDF5...\n")

	## double center the dosage matrix by gene and individual IN CHUNKS
	gene_array = expression_promoter_df['Gene'].values
	ind_array = expression_promoter_df['Individual'].values

	n_rows = dosage_df.shape[0]
	n_cols = dosage_df.shape[1]
	
	# Define your output file
	hdf5_X_double_centered = output + "_kmer_cluster_double_centered.h5"
	
	with h5py.File(hdf5_X_double_centered, "w") as f:
		dset = f.create_dataset("X_double_centered", shape=(n_rows, n_cols), dtype='float32')
		
		chunk_size = 1000 # Safely processes 1,000 kmers at a time
		
		for i in range(0, n_cols, chunk_size):
			# 1. Slice the specific 1,000 columns
			kmer_chunk_names = kmer_clusters[i : i + chunk_size]
			sparse_chunk = dosage_df[kmer_chunk_names]
			
			# 2. Convert ONLY this small chunk to dense RAM
			dense_chunk = sparse_chunk.sparse.to_dense()
			
			# 3. Calculate means just for this chunk
			kmer_mu = dense_chunk.mean()
			kmer_gene_mean = dense_chunk.groupby(gene_array,sort = False).transform('mean')
			kmer_ind_mean = dense_chunk.groupby(ind_array,sort = False).transform('mean')
			
			# 4. Double center the chunk
			centered_chunk = dense_chunk - kmer_gene_mean - kmer_ind_mean + kmer_mu
			
			# 5. Write the result directly into the HDF5 file and cast to float32
			dset[:, i : i + chunk_size] = centered_chunk.values.astype(np.float32)
			
			print(f"Processed columns {i} to {min(i + chunk_size, n_cols)} out of {n_cols}")
	
	print("\nFinished calculating the double centered kmer matrix. Saved safely to disk!\n")

	## next calculate the the gene baseline, gene-specific promoter features, and X_baseline

	unique_genes_sorted, idx = np.unique(expression_promoter_df['Gene'].values,return_index=True)
	unique_genes_original_order = unique_genes_sorted[np.argsort(idx)]

	num_genes = len(unique_genes_original_order)

	hdf5_X_baseline = output + "_kmer_cluster_baseline.h5"
	
	with h5py.File(hdf5_X_baseline, "w") as f:
		dset = f.create_dataset("X_baseline", shape=(num_genes, n_cols), dtype='float32')
		
		chunk_size = 1000 # Safely processes 1,000 kmers at a time

		for i in range(0, n_cols, chunk_size):
			# 1. Slice the specific 1,000 columns
			kmer_chunk_names = kmer_clusters[i : i + chunk_size]
			sparse_chunk = dosage_df[kmer_chunk_names]
			
			# 2. Convert ONLY this small chunk to dense RAM
			dense_chunk = sparse_chunk.sparse.to_dense()
			
			# 3. Calculate means just for this chunk
			X_baseline_chunk = dense_chunk.groupby(gene_array,sort = False).mean()
			X_baseline_centered_chunk = X_baseline_chunk - X_baseline_chunk.mean()
			
			# 4. Write the result directly into the HDF5 file and cast to float32
			dset[:, i : i + chunk_size] = X_baseline_centered_chunk.values.astype(np.float32)
			
			print(f"Processed columns {i} to {min(i + chunk_size, n_cols)} out of {n_cols}")
	
	print("\nFinished calculating the baseline centered kmer matrix. Saved safely to disk!\n")

	promoter_feature_cols = [col for col in promoter_feat_df.columns if col != "Allele"]
	cols_to_aggregate = ['alpha_i'] + promoter_feature_cols
	gene_level_metadata = expression_promoter_df.groupby('Gene',sort = False)[cols_to_aggregate].mean()
	
	if gene_level_metadata.index.to_list() != unique_genes_original_order.tolist():
		sys.exit("ERROR: The gene names in the collapsed metadata do not match the gene names in the X_baseline matrix. Please double check!")
	
	gene_level_metadata = gene_level_metadata.reset_index()	
	
	print("Finished calculating the gene baseline expression and gene-level promoter features.\n")

	return (expression_promoter_df,gene_level_metadata,promoter_feature_cols)






















