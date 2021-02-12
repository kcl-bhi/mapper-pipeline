# Title:        Prepare pipeline inputs for different parameter combinations
# Author:       Ewan Carr (@ewancarr)
# Updated:      2021-02-10

import os
import shutil
import pickle
import csv
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_multilabel_classification
import gower
import urllib.request

# Check if input dataset is available
data_avail = os.path.exists('input.csv')

# Load dataset if available; otherwise simulate
if data_avail:
    data = pd.read_csv('input.csv').dropna()
else:
    sim = make_multilabel_classification(n_samples = 430,
                                         n_features = 137,
                                         n_classes = 8)
    data = pd.DataFrame(sim[0])
    data.columns = ['X' + str(i + 1) for i in data.columns]
    data['fila'] = np.random.uniform(-5, 5, len(data.index))
    data['filb'] = np.random.uniform(-5, 5, len(data.index))
    data['ycont'] = np.random.uniform(30, 80, len(data.index))
    data['ybin'] = data['ycont'] > 60

# Identify categorical items

"""
For computing the Gower matrix (this file) and descriptive statistics
(later) we need to specify which variables in the input dataset should
be treated as categorical. This can be specified with a CSV file
containing the columns 'index' and 'categorical':

index:          A column of variable names matching those in 'input.csv'
categorical:    A column of 0 or 1 indicating wether each variable should
                be treated as categorical.
"""

if data_avail:
    categ = pd.read_csv('categorical_items.csv')[['index', 'categorical']]
else:
    categ = pd.DataFrame({'index': list(data),
                          'categorical': np.concatenate([np.repeat(1, 137),
                                                         np.array([0, 0, 0, 1])])})
    categ.to_csv('categorical_items.csv')

is_categ = categ['categorical'].values == 1

# Compute Gower distance matrix
gmat = gower.gower_matrix(data, cat_features=is_categ)


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
fil['fila'] = data['fila'].values
fil['filb'] = data['filb'].values
fil['ycont'] = data['ycont'].values

# Create combinations of the above
for j in ['fila', 'filb', 'ycont']:
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
out = 'outputs'

# Create folder for inputs; delete if exists already
if os.path.isdir(inp):
    shutil.rmtree(inp) 
os.mkdir(inp)

if not os.path.isdir(out):
    os.mkdir(out)

for i, p in enumerate(params, 1):
    gid = f'{i:04}'
    # Save params as pickle
    with open(os.path.join(inp, gid + '.pickle'), 'wb') as f:
        pickle.dump(p, f)
    # Add to index CSV
    with open(os.path.join(inp, 'index.csv'), 'a') as f:
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
data['ybin'].to_csv(os.path.join(inp, 'ybin.csv'), index=False, header=False)
data['ycont'].to_csv(os.path.join(inp, 'ycont.csv'), index=False, header=False)

# Baseline clinical features (excluding any outcomes) -------------------------
to_drop = ['fila', 'filb', 'ycont', 'ybin']
data.drop(columns=to_drop, axis=1) \
    .to_csv(os.path.join(inp, 'X.csv'), index=False, header=True)

# Gower distance matrix
pd.DataFrame(gmat).to_csv(os.path.join(inp, 'gower.csv'),
                          index=False,
                          header=False)

