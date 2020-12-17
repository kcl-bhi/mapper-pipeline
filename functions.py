import numpy as np
import matplotlib.pyplot as plt
import statmapper as stm
from sklearn_tda import MapperComplex
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def gini(p):
    return(1 - (1 - p)**2 - (p)**2)


def matches(item, comparison):
    comparison = ["{:10.10f}".format(i) for i in comparison]
    return("{:10.10f}".format(item) in comparison)


def extract_xy(d):
    x = np.array([i[1][0] for i in d])
    y = np.array([i[1][1] for i in d])
    return([x, y])


def compare_features(M, computed, ax=None):
    ax = ax or plt.gca()
    x0, y0 = extract_xy(M.compute_persistence_diagrams()[0])
    x1, y1 = extract_xy(M.compute_persistence_diagrams()[0])

    x_orig = np.hstack([x0, x1])
    y_orig = np.hstack([y0, y1])

    x_comp, y_comp = extract_xy(computed)

    color = []
    for i in range(len(x_orig)):
        if matches(x_orig[i], x_comp) and matches(y_orig[i], y_comp):
            color.append('red')
        else:
            color.append('grey')

    # Make the plot
    dag = np.arange(np.min(x_orig), np.max(x_orig))
    _ = ax.scatter(x_orig, y_orig, c=color)
    _ = ax.plot(dag, dag)
    _ = ax.set_xlabel('Birth')
    _ = ax.set_ylabel('Death')
    return(_)


def convert_gain(gain_aysadi):
    return(1-(1/gain_aysadi))


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃          Functions to count/summarise features in Mapper graphs           ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def dg(M, X=None):
    """Describe a Mapper graph"""
    # Get total number of rows in input data
    if X is not None:
        print(np.shape(X)[0], 'rows in input data')
    # Get total number of nodes; unique members
    members = np.array([])
    for k, v in M.node_info_.items():
        members = np.hstack([members, v['indices']])
    print(len(M.node_info_), 'nodes')
    print(len(np.unique(members)), 'unique members')
    print('┌────────┬─────────────┐')
    print('│ ID     │ N members   │')
    print('├────────┼─────────────┤')
    for k, v in M.node_info_.items():
        print('│ ', k, ' ' * (4 - len(str(k))),
              '│', v['size'], ' ' * (10 - len(str(v['size']))), '│')
    print('└────────┴─────────────┘')


def count_features(M):
    feature_types = ['downbranch', 'upbranch', 'connected_component', 'loop']
    n_feat, n_sig = 0, 0
    for topo in feature_types:
        n_feat = n_feat + len(M['dgm'][topo])
        n_sig = n_sig + len(M['sdgm'][topo])
    res = [n_feat, n_sig]
    for i in ['dgm', 'sdgm']:
        for j in feature_types:
            res.append(len(M[i][j]))
    return(res)


def representative_features(M, confidence, bootstrap, inp):
    features = ['downbranch', 'upbranch',
                'connected_component', 'loop']
    for topo in features:
        # Compute and save representative features
        if 'dgm' not in M.keys():
            M['dgm'], M['bnd'] = {}, {}
        dgm, bnd = stm.compute_topological_features(M=M['map'],
                                                    func=M['fil'][:, 0],
                                                    func_type="data",
                                                    topo_type=topo,
                                                    threshold=confidence)
        M['dgm'][topo] = dgm
        M['bnd'][topo] = bnd
        # Run bootstrap for representative features
        if 'sdgm' not in M.keys():
            M['sdgm'], M['sbnd'] = {}, {}
        sdgm, sbnd = stm.evaluate_significance(dgm=M['dgm'][topo],
                                               bnd=M['bnd'][topo],
                                               X=M['X'],
                                               M=M['map'],
                                               func=M['fil'],
                                               params=M['params'],
                                               topo_type=topo,
                                               threshold=confidence,
                                               N=bootstrap,
                                               input=inp)
        M['sdgm'][topo] = sdgm
        M['sbnd'][topo] = sbnd
    return(M)


def identify_membership(pick):
    """
    Given a set of input parameters, re-run Mapper and create a dataframe that
    identifies which participants belong to which features. Note that this
    function retains significant features only.
    """
    # Re-run Mapper
    mapper = MapperComplex(**pick['params']).fit(pick['X'])
    # Get nodes for each feature
    nodes = {k: v['indices'] for k, v in mapper.node_info_.items()}
    # Get node memberships for each participant
    pid = {}
    for i in range(np.shape(pick['X'])[0]):
        pid[i] = {}
        pid[i]['nodes'] = []
        for k, v in nodes.items():
            if i in v:
                pid[i]['nodes'].append(k)
    # Get feature memberships
    feature_types = ['downbranch', 'upbranch', 'connected_component', 'loop']
    for i in feature_types:
        id = 1
        for f in pick['sbnd'][i]:
            if not all([len(i) == 0 for i in pick['sbnd'][i]]):
                # Only compute feature types that exist
                feature = i[0].upper() + str(id)
                id += 1
                for pk, pv in pid.items():
                    if len(set(f).intersection(pv['nodes'])) > 0:
                        pid[pk][feature] = True
                    else:
                        pid[pk][feature] = False
    membership = pd.DataFrame.from_dict(pid, orient='index')
    return(membership)


def remove_small_features(membership, prop=0.1):
    for col in list(membership)[1:]:
        y_prop = np.mean(membership[col])
        if (y_prop > (1-prop)) or (y_prop < prop):
            membership = membership.drop(col, axis=1)
    return(membership)


def count_sig(sbnd):
    tot = 0
    for k, v in sbnd.items():
        tot += len(v)
    return(tot)


def get_imps(X, clf):
    imp = pd.DataFrame(clf.feature_importances_,
                       list(X),
                       columns=['imp'])
    return(imp.sort_values(by='imp',
                           ascending=False)[:10].reset_index())


def get_scores(X, y, clf):
    scores = cross_val_score(clf,
                             X.values,
                             y,
                             cv=3,
                             scoring='roc_auc')
    return({'max': np.max(scores),
            'mean': np.mean(scores)})


def predict_feature_membership(X, membership, XGBoost=True):
    """
    Predicts feature membership with RF or XGBoost based on baseline clinical
    variables, using 3-fold CV. Returns max/mean AUC
    scores AND the top 10 most influential features.
    """
    outcomes = membership.drop(['nodes'], axis=1)
    results = {}
    for c, y in outcomes.iteritems():
        i = {}
        if XGBoost:
            clf = XGBClassifier(random_state=42).fit(X.values, y)
            i['auc'] = get_scores(X, y, clf)
            i['imp'] = get_imps(X, clf)
        else:
            clf = RandomForestClassifier(n_estimators=100,
                                         max_depth=2,
                                         random_state=42).fit(X.values, y)
            i['auc'] = get_scores(X, y, clf)
            i['imp'] = get_imps(X, clf)
        results[c] = i
    return(results)


def create_table_of_predictions_results(list_of_solutions):
    # Get best-fitting feature for each graph
    for i in list_of_solutions:
        if i['prediction'] == "No features large enough":
            i['summary'] = [np.nan, np.nan]
        else:
            # Pick the best-fitting feature, if graph has multiple significant
            # topological features
            highest = 0
            for k, v in i['prediction'].items():
                if v['auc']['mean'] > highest:
                    highest = v['auc']['mean']
                    feat = k
            i['summary'] = [feat, highest]
    # Produce a table of best-fitting features, across all graphs
    results = {k: v for k, v in enumerate(list_of_solutions)}
    perf = pd.DataFrame([(k,
                          v['fil_lab'],
                          v['summary'][0],
                          v['summary'][1]) for k, v in results.items()],
                        columns=['model', 'filter', 'feature', 'mean_auc'])
    perf = perf.sort_values(by=['mean_auc'], ascending=False).dropna()
    return(perf)
