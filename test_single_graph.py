# Title:        Test single Mapper graph on Rosalind
# Author:       Ewan Carr
# Started:      2020-05-06

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn_tda import MapperComplex
from functions import (representative_features,
                       identify_membership,
                       count_features)
debug = False

'''
This script three inputs:
    1. A index number (0001-9999).
    2. An input directory (e.g. 'inputs')
    3. An output directory (e.g. ('outputs')

It then:
    1. Loads the corresponding data/parameters (e.g. from inputs/0001.pickle).
    2. Runs Mapper with these parameters.
    3. Identifies significant, representative topological features.
    4. Extracts information on the number and type of features.
'''

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                            STEP 1: Load inputs                            ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

print("Loading inputs...")
if debug:
    i = '0333'
    inputs = 'inputs/'
    outputs = 'outputs/'
else:
    i = sys.argv[1]
    inputs = sys.argv[2] + '/'
    outputs = sys.argv[3] + '/'

# Load inputs/datasets
current = pickle.load(open(inputs + i + '.pickle', 'rb'))
X = pd.read_csv(inputs + 'X.csv')
ybin = pd.read_csv(inputs + 'ybin.csv', header=None, names=['hdremit.all'])
gower = pd.read_csv(inputs + 'gower.csv', header=None)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                            STEP 2: Run Mapper                             ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

print("Running Mapper...")
fil_label, fil = current['fil']
if fil.ndim == 1:
    fil = fil.reshape(-1, 1)
n_filters = np.shape(fil)[1]
res, clust = current['res']
gain = current['gain']
params = {'inp': 'distance matrix',
          'filters': fil,
          'filter_bnds': np.array(np.nan * n_filters),
          'colors': fil,
          'resolutions': np.array([res] * n_filters),
          'gains': np.array([gain] * n_filters)}
if clust == 'auto':
    m = MapperComplex(**params).fit(gower.values)
else:
    m = MapperComplex(**params,
                      clustering=clust).fit(gower.values)
M = {'fil': fil,
     'X': gower.values,
     'map': m,
     'params': params}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                   STEP 3: Identify significant features                   ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

print("Identifying significant features...")
rep_feat = representative_features(M,
                                   confidence=0.90,
                                   bootstrap=100,
                                   inp='distance matrix')
membership = identify_membership(rep_feat)
print(np.shape(membership))

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                     STEP 4: Extract required outputs                      ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

feature_counts = count_features(rep_feat)
feature_labels = ['n_feat',
                  'n_sig',
                  'downbranch',
                  'upbranch',
                  'conn',
                  'loop',
                  'sig_downbranch',
                  'sig_upbranch',
                  'sig_conn',
                  'sig_loop']
results = {}
for c, l in zip(feature_counts, feature_labels):
    results[l] = c
results['n_nodes'] = len(M['map'].node_info_.items())
results['memb'] = membership
results['params'] = current

with open(os.path.join(outputs, i + '.pickle'), 'wb') as f:
    pickle.dump(results, f)
