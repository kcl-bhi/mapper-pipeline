# Title:        Prepare pipeline inputs for different parameter combinations
# Author:       Ewan Carr
# Updated:      2020-07-03

import os
import pickle
import csv
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import gower
import urllib.request


# Load dataset
gendep = pd.read_csv('baseline.csv').dropna()

# Compute Gower distance matrix
gmat = gower.gower_matrix(gendep)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                         Define parameter settings                         ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Define filters ==============================================================
fil = {}

# MDS, with Gower distance matrix, first two components
fil['mds'] = MDS(n_components=2,
                 dissimilarity='precomputed').fit_transform(gmat)

# MADRS, BDI, PRS
fil['madrs'] = gendep['madrs.total'].values
fil['mdper'] = gendep['mdpercadj'].values
fil['prs'] = gendep['prs'].values

# Create combinations of the above
for j in ['madrs', 'prs', 'mdper']:
    fil['mds' + '_' + j] = np.column_stack([fil['mds'], fil[j]])

# Define clustering algorithm =================================================
cluster = [DBSCAN(eps=i, metric='precomputed')
           for i in[0.5, 1, 5, 10, 50, 100]]
cluster.append(AgglomerativeClustering(affinity='precomputed',
                                       linkage='average'))
cluster.append('auto')

# Define resolutions ==========================================================
resolutions = [(res, c)
               for res in [1, 3, 5, 10, 30, 50]
               for c in cluster]
resolutions.append((np.nan, None))


# Create dictionary containing all combinations of input parameters ===========
params = [{'fil': f,
           'res': r,
           'gain': gain}
          for f in fil.items()
          for r in resolutions
          for gain in [0.1, 0.2, 0.3, 0.4]]

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                 Generate input files for all combinations                 ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Set folder to store input files
inp = 'inputs'

# Delete existing input files
for f in os.listdir(inp):
    os.unlink(os.path.join(inp, f))

for i, p in enumerate(params, 1):
    gid = f'{i:04}'
    # Save params as pickle
    with open(inp + gid + '.pickle', 'wb') as f:
        pickle.dump(p, f)
    # Add to index CSV
    with open(inp + 'index.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([gid,
                         str(p['fil'][0]),
                         str(p['gain']),
                         str(p['res'][0]),
                         str(p['res'][1])])
    # Add to job list for GNU parallel
    with open('jobs', 'a+') as f:
        f.write('python test_single_graph.py ' + gid + ' inputs outputs\n')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                      Export other required datasets                       ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Binary 12-week outcome ------------------------------------------------------
gendep['hdremit.all'].to_csv(inp + 'y.csv', index=False, header=False)
gendep['mdpercadj'].to_csv(inp + 'mdpercadj.csv', index=False, header=False)

# Baseline clinical features (excluding any outcomes) -------------------------
gendep.drop(columns=['centreid',
                     'hdremit.all',
                     'mdpercadj']).to_csv(inp + 'X.csv',
                                          index=False,
                                          header=True)
# Gower distance matrix
pd.DataFrame(gmat).to_csv(os.path.join(inp,
                                       'gower.csv'),
                          index=False,
                          header=False)
