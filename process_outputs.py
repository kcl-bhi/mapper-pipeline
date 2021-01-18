# Title:        Process outputs each Mapper graph
# Author:       Ewan Carr

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn_tda import MapperComplex
from functions import gini
from statsmodels.stats.proportion import proportion_confint
import statsmodels.stats.api as sms
from joblib import Parallel, delayed, dump, load
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
import xgboost as xgb
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import scale
import xlsxwriter
import tda.helpers as th
import tda.mapper as tm
import tda.plotting as tp


# Load baseline/outcome variables ---------------------------------------------
X = pd.read_csv(os.path.join('inputs', 'X.csv'))
gower = pd.read_csv(os.path.join('inputs', 'gower.csv'), header=None)
ycont = pd.read_csv(os.path.join('inputs', 'ycont.csv'),
                    header=None, names=['ycont'])
ybin = pd.read_csv(os.path.join('inputs', 'ybin.csv'),
                   header=None, names=['ybin'])

# Load all Mapper graphs ------------------------------------------------------
outdir = 'outputs'
res_all = {}
for f in tqdm(os.listdir(outdir)):
    p = os.path.join(outdir, f)
    res_all[f[0:4]] = pickle.load(open(p, 'rb'))

# Remove graphs with no significant features ----------------------------------
print('All graphs:', len(res_all))
res = {k: v for k, v in res_all.items() if v['n_sig'] > 0}
print('Graphs with >0 significant features:', len(res))

# Split graphs into separate features -----------------------------------------
features = []
for k, v in res.items():
    feat = v['memb'].columns[1:]
    for f in feat:
        other_features = [x for x in feat
                          if x != f
                          if v['memb'][x].sum() > 21]
        if v['params']['res'][1] == 'auto':
            clust = 'DBSCAN (auto)'
        else:
            clust = type(v['params']['res'][1]).__name__
        features.append({'graph': k,
                         'feat': f,
                         'memb': v['memb'][f],
                         'other': v['memb'][other_features],
                         'n': v['memb'][f].sum(),
                         'fil': v['params']['fil'][0],
                         'res': v['params']['res'][0],
                         'gain': v['params']['gain'],
                         'cluster': clust})
print('Number of features =', len(features))

# Remove features with <5% or >95% of the sample ------------------------------
features = [f for f in features
            if (f['n'] > (0.05 * len(f['memb'])))
            if (f['n'] < (0.95 * len(f['memb'])))]
print('Number of features =', len(features))

# Calculate homogeneity -------------------------------------------------------

# Homogeneity for sample
gini_samp = gini(ybin.mean())[0]
std_samp = ycont.std()[0]

# Homogeneity for each feature
for f in tqdm(features):
    # Binary outcome (ybin)
    p = pd.concat([f['memb'], ybin], axis=1) \
        .groupby(f['feat'])['ybin'] \
        .mean()[1]
    gini_feat = gini(p)
    gini_pct = ((gini_samp - gini_feat) / gini_samp) * 100
    # Continuous outcome (ycont)
    std_feat = pd.concat([f['memb'],
                          ycont],
                         axis=1).groupby(f['feat'])['ycont'].std()[1]
    std_pct = ((std_samp - std_feat) / std_samp) * 100
    # Store
    f['homog'] = {'gini': gini_feat,
                  'gini_pct': gini_pct,
                  'std': std_feat,
                  'std_pct': std_pct}


# Calculate differences -------------------------------------------------------
def calculate_distances(f, y):
    d = {}
    # Get mean/proportion of sample and feature
    d_feat = float(y[f['memb']].mean())
    d_restofsamp = float(y[~f['memb']].mean())
    d['mean_feat'] = d_feat
    d['mean_samp'] = d_restofsamp
    # Calculate difference
    d['diff_samp'] = float(d_feat - d_restofsamp)
    # Calculate difference to each other feature
    for o in f['other']:
        d['diff_' + o] = float(d_feat - y[f['other'][o]].mean())
    # Calculate maximum distance to all other features
    dmax = 0
    for k in ['diff_' + x for x in list(f['other'])]:
        if abs(d[k]) > dmax:
            dmax = abs(d[k])
    d['diff_max'] = dmax
    return(d)


def get_dist(f):
    dist = {}
    dist['ybin'] = calculate_distances(f, ybin)
    dist['ycont'] = calculate_distances(f, ycont)
    dist['n_other'] = len(list(f['other']))
    f['dist'] = dist
    return(f)


features = Parallel(n_jobs=8)(delayed(get_dist)(f) for f in features)
dump(features, 'features.joblib')

# Remove duplicate features ---------------------------------------------------
for f, i in zip(features, range(len(features))):
    f['n_other'] = f['dist']['n_other']
    f['max_ybin'] = f['dist']['ybin']['diff_max']
    f['max_ycont'] = f['dist']['ycont']['diff_max']
    f['gini'] = f['homog']['gini']
    f['gini_pct'] = f['homog']['gini_pct']
    f['std'] = f['homog']['std']
    f['std_pct'] = f['homog']['std_pct']
    f['id'] = i
feature_summary = pd.DataFrame.from_dict(features)[['id',
                                                    'graph', 'feat',
                                                    'n', 'gini', 'std',
                                                    'max_ybin',
                                                    'max_ycont']]
unique = feature_summary.drop_duplicates(subset=['n', 'gini', 'std'])
unique_features = [f for f in features if f['id'] in unique['id']]
print('features:', len(features))
print('unique:', len(unique_features))

# Function to create summary table listing all features -----------------------
def tab(lof, sort=None, ascending=False):
    cols = ['id', 'graph', 'feat', 'n', 'gini', 'gini_pct',
            'std', 'std_pct', 'max_ybin', 'max_ycont', 'n_other',
            'fil', 'res', 'gain', 'cluster']
    cols = cols + ['overlap' + str(i) for i in range(5)]
    sr = []
    for l in lof:
        row = []
        for k in cols:
            row.append(l.get(k))
        sr.append(row)
    results_table = pd.DataFrame(sr, columns=cols)
    to_round = ['gini', 'gini_pct', 'std',
                'std_pct', 'max_ybin', 'max_ycont']
    results_table[to_round] = results_table[to_round].round(2)
    results_table.set_index(['graph', 'feat'], inplace=True)
    results_table.columns = pd.MultiIndex.from_product([['summary'],
                                                        results_table.columns])
    if sort:
        return(results_table.sort_values(('summary', sort),
                                         ascending=ascending))
    else:
        return(results_table)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                           Select top N features                           ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Outcome 1: continuous -------------------------------------------------------
gini = [f['homog']['gini'] for f in features]
gini_pct = [f['homog']['gini_pct'] for f in features]
# NOTE: gini_pct refers to the percentage *reduction* in Gini

# Sort by homogeneity
top20 = np.sort(gini_pct)[::-1][np.min([len(gini_pct) - 1, 20])]
top_ybin = [f for f in features if f['homog']['gini_pct'] >= top20]

# Outcome 2: binary -----------------------------------------------------------
std = [f['homog']['std'] for f in features]
std_pct = [f['homog']['std_pct'] for f in features]

# Sort by homogeneity
top20 = np.sort(std_pct)[::-1][np.min([len(gini_pct) - 1, 20])]
top_ycont = [f for f in features if f['homog']['std_pct'] >= top20]

tab(features, 'gini_pct')
tab(top_ybin, 'gini_pct')
tab(top_ycont, 'std_pct')

# Make table summarising all features -----------------------------------------
af = tab(features, 'gini_pct')
af.columns = [i[1] for i in af.columns]
af['gini_samp'] = gini_samp
af['std_samp'] = std_samp
af.to_excel('all_features.xlsx', merge_cells=False)

# Tabulate distances for each feature -----------------------------------------
diffs = []
for f in features:
    tf = []
    for o in ['ybin', 'ycont']:
        d = pd.DataFrame({'graph': f['graph'],
                          'feat': f['feat'],
                          **f['dist'][o]}, index=[0])
        d.set_index(['graph', 'feat'], inplace=True)
        d.columns = pd.MultiIndex.from_product([[o], d.columns])
        tf.append(d)
    diffs.append(pd.concat(tf, axis=1))
diffs = pd.concat(diffs, axis=0)


# Get mean differences in baseline clinical variables -------------------------
def get_stat(x):
    if len(np.unique(x)) > 2:
        low, high = sms.DescrStatsW(x).tconfint_mean()
        return('{:.2f}'.format(np.mean(x)) + ' [' +
               '{:.2f}'.format(low) + ', ' +
               '{:.2f}'.format(high) + ']')
    else:
        x = x - (np.max(x) - 1)
        low, high = proportion_confint(sum(x), len(x))
        return('{:.1f}'.format(np.mean(x) * 100) + '% [' +
               '{:.1f}'.format(low * 100) + ', ' +
               '{:.1f}'.format(high * 100) + ']')


def chi2(df, col1, col2):
    return chi2_contingency([
        [
            len(df[(df[col1] == cat) & (df[col2] == cat2)])
            for cat2 in range(int(df[col1].min()), int(df[col1].max()) + 1)
        ]
        for cat in range(int(df[col2].min()), int(df[col2].max()) + 1)
    ])


def get_fi(X, y):
    clf = xgb.XGBClassifier(eval_metric='logloss',
                            use_label_encoder=False)
    clf.fit(scale(X), y)
    auc = np.mean(cross_val_score(clf, X, y, cv=10,
                                  scoring=make_scorer(roc_auc_score)))
    return((auc,
            pd.Series(clf.feature_importances_,
                      name='xgb_fi', index=list(X))))


def draw_graph(i, path):
    print('Running Mapper for ' + i + '...')
    inputs = res_all[i]['params']
    fil_label, fil = inputs['fil']
    if fil.ndim == 1:
        fil = fil.reshape(-1, 1)
    n_filters = np.shape(fil)[1]
    res, clust = inputs['res']
    print(fil_label, np.shape(fil))
    gain = inputs['gain']
    params = {'inp': 'distance matrix',
              'filters': fil,
              'filter_bnds': np.array(np.nan * n_filters),
              'colors': fil,
              'resolutions': np.array([res] * n_filters),
              'gains': np.array([gain] * n_filters)}
    if clust == 'auto':
        m = MapperComplex(**params).fit(gower.values)
    else:
        m = MapperComplex(**params, clustering=clust).fit(gower.values)
    M = {'fil': fil,
         'X': gower.values,
         'map': m,
         'params': params}
    bootstrap = th.representative_features(M,
                                           confidence=0.9,
                                           bootstrap=100,
                                           inp="distance matrix")
    M['bootstrap'] = bootstrap
    P = tp.graph_features(M['bootstrap'])
    P.graph_attr['overlap'] = 'scale'
    P.graph_attr['font'] = 'Calibri'
    P.graph_attr['sep'] = 1
    P.graph_attr['splines'] = 'true',
    P.draw(path, prog="neato")


# Combine distances with feature summaries ------------------------------------
opts = {'ybin': {'top': top_ybin,
                 'sort': 'gini_pct'},
        'ycont': {'top': top_ycont,
                  'sort': 'std_pct'}}

# Load CSV specifying which baseline variables are categorical ----------------
categ = pd.read_csv('categorical_items.csv')[['index', 'categorical']]
categ.set_index('index', inplace=True)
categ = categ['categorical'].apply(lambda x: True if x == 1 else False)

# Get required summaries for each of the top-ranked features ------------------
for k, v in opts.items():
    # Summary of this set of features
    v['sumstat'] = tab(v['top'], v['sort']) \
        .merge(diffs,
               left_index=True,
               right_index=True,
               how='inner').round(2).reset_index().fillna('--') \
        .sort_values(('summary', v['sort']), ascending=False)
    # For each feature in this top 20
    for f in tqdm(v['top']):
        pd.concat([f['memb'], ybin, ycont], axis=1)
        # Get means/proportions for baseline variables
        baseline_diffs = pd.concat([f['memb'], X], axis=1) \
            .groupby(f['feat']) \
            .agg(get_stat).T
        # Add means of outcomes (ybin, ycont)
        oc = pd.concat([f['memb'], ybin, ycont], axis=1)
        oc = oc.groupby(oc.iloc[:, 0])[['ybin', 'ycont']] \
            .agg(get_stat).T
        baseline_diffs = pd.concat([baseline_diffs, oc])
        # Get two-sample KS test (continuous)/ chi-square test (categorical)
        XM = pd.concat([X, f['memb'], ybin, ycont], axis=1)
        XM.rename(columns={f['feat']: 'memb'}, inplace=True)
        for i in XM:
            if i != 'memb':
                if categ[i]:
                    c, p = chi2_contingency(pd.crosstab(XM[i],
                                                        XM['memb']))[0:2]
                    cell = 'Chi2 = {:.5f}; pval={:.5f}'.format(c, p)
                    baseline_diffs.loc[i, 'ks_chi'] = cell
                else:
                    d1 = XM[~XM.memb][i].values
                    d2 = XM[XM.memb][i].values
                    stat, pval = ks_2samp(d1, d2)
                    cell = 'KS = {:.5f}; pval={:.5f}'.format(stat, pval)
                    baseline_diffs.loc[i, 'ks_chi'] = cell
        # Get XGB AUC and feature importances
        auc, fi = get_fi(X, f['memb'])
        f['auc'] = auc
        # Combine with baseline differences
        fi = pd.concat([fi], axis=1)
        fi.columns = ['fi']
        with_fi = baseline_diffs \
            .merge(fi,
                   left_index=True,
                   right_index=True,
                   how='left').reset_index()
        f['with_fi'] = with_fi
        # Get list of feature members (based on row number of input dataset)
        f['memb_col'] = pd.DataFrame({'row ID': f['memb'].index[f['memb']]})
    # Sort the features
    v['top'] = sorted(v['top'], key=lambda k: k[v['sort']], reverse=True)

# Generate Excel documents ----------------------------------------------------
if not os.path.exists('examples'):
    os.mkdir('examples')
for k, v in opts.items():
    wb = xlsxwriter.Workbook(os.path.join('examples', 
                                          k + '.xlsx'),
                             {'nan_inf_to_errors': True})
    bigfont = wb.add_format()
    bigfont.set_font_size(18)
    bold = wb.add_format()
    bold.set_bold()
    for n, f in enumerate(v['top']):
        sid = '_'.join([f['graph'], f['feat']])
        ws = wb.add_worksheet(str(n + 1) + ' | ' + sid)
        ws.write(0, 0, sid, bigfont)
        ws.set_row(0, 25)
        for i, c in enumerate(f['with_fi']):
            ws.write_column(3, i, f['with_fi'][c])
        ws.write_row(2, 0, list(with_fi))
        # Write AUC
        ws.write('E2', 'AUC: ' + '{:.10f}'.format(f['auc']), bold)
        # Add feature summaries
        ss = v['sumstat']
        for s, r in zip(['summary', 'ybin', 'ycont'],
                        [2, 25, 48]):
            ws.write_column(r + 1, 7, ss['graph'])
            ws.write_column(r + 1, 8, ss['feat'])
            for i, col in enumerate(ss[(s, )]):
                ws.write_column(r + 1, 8 + i, ss[s][col])
            ws.write_row(r, 8, list(ss[(s, )]))
        # Add formatting/labels
        ws.conditional_format('E4:E141', {'type': '3_color_scale'})
        ws.set_column('A:E', 15)
        ws.set_column('D:E', 25)
        ws.write('A2', 'Baseline differences', bold)
        ws.write('H2', 'Feature summaries', bold)
        ws.write('H25', 'ybin: Distances to other features, means',
                 bold)
        ws.write('H48', 'ycont: Distance to other features, means', bold)
        ws.write_row(2, 0,
                     [' ', 'Non-member',
                      'Member', 'KS/Chi2',
                      'XGB gain'],
                     bold)
        # Add Mapper graph
        figdir = os.path.join('figures', 'for_excel')
        if not os.path.exists(figdir):
            os.mkdir(figdir)
        p = os.path.join(figdir, f['graph'] + '.png')
        if not os.path.exists(p):
            draw_graph(f['graph'], p)
        ws.insert_image('H71', p)
        # Add column with IDs of cluster members
        ws.write(2, 28, 'Cluster members (based on row numbers of input dataset')
        ws.write_column(3, 28, f['memb_col'].values)
    wb.close()
