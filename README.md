# K-PROB (Kmer-based In silico promoter bashing)

<img width="1294" height="425" alt="image" src="https://github.com/user-attachments/assets/b4bbb33e-d617-4a60-a062-d4cc4ae63cd0" />

### Citation
Wei, Wei, et al. "Natural variation in regulatory code revealed through Bayesian analysis of plant pan-genomes and pan-transcriptomes." bioRxiv (2025): 2025-10.

### Overview
***K-PROB** is a computational tool that learns from intraspecies promoter sequence and gene expression variation in pan-genomes and pan-transcriptomes to identify CREs controlling gene expression. It deploys a k-mer-based Bayesian variable selection framework to prioritize causal variable identification

### Installation
We highly recommend users to use mamba for installation

```mamba env create -f environment.yml```

```conda activate krispr```

### Command line of K-PROB


```
usage: kprob.py [-h] [-f SEQUENCE] [-k K] [-g GAP] [-l CUTOFF] [-t TASK] [-u] [-w WORDSIZE] [-s SIMILARITY] [-x GENO] [-c COVAR] [-y PHENO]
                 [-m MODEL] [-s0 S0] [-b PI_B] [-n NUM] [-v VERBOSE] [-o OUTPUT]

options:
  -h, --help     show this help message and exit
  -f SEQUENCE    the multi-fasta file
  -k K           size of the kmer
  -g GAP         the number of nucleotide gap between 2 kmers
  -l CUTOFF      a separate DM matrix with the minimum occurrence frequency for a kmer/kmer cluster to be included in the mapping (default:0.01)
  -t TASK        count | mapping
  -u             output the unique kmer dosage matrix (default: false)
  -w WORDSIZE    the wordsize for cd-hit-est
  -s SIMILARITY  the similarity cutoff for cd-hit-est
  -x GENO        the input matrix (X) for the mapping step
  -c COVAR       the covariates (C) for the mapping step
  -y PHENO       the response variable for the mapping step
  -m MODEL       the statistical model for kmer effect estimation. Krispr offers two spike priors 1 (default): small effect around 0; 2: point mass
                 at 0
  -s0 S0         the proportion of phenotypic variation explained by background kmers
  -b PI_B        pi_b for the beta distribution
  -n NUM         the number of threads for kmer counting / MCMC chains. Recommend at least 5
  -v VERBOSE     verbose levels 0: no stdout; 1: convergence and minimal stdout; 2: per MCMC iteration stdout
  -o OUTPUT      the prefix of the output files

```

### Usage Step 1
This is the **first** step of K-PROB, it takes a fasta file and identifies unique kmers and then cluster them and generate a kmer cluster dosage matrix

```python3 krispr.py -t count -f promoter.fa -k kmer_size  -o output```

### Usage Step 2
This is the **second** step of K-PROB, it takes the kmer cluster dosage matrix and an expression vector as input files, and run the regression model to identify putitative causal kmer clusters

```python3 krispr.py -t mapping -x DosageMatrix.txt -y expression.txt -o output```

### Contact Info
Wei Wei (doublevi123_at_gmail.com) / Xing Wu (wuxingtom_at_gmail.com)
