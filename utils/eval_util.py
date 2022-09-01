import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from csv import DictWriter
import os
from copy import deepcopy

def eval_scores(pred, truth):

    try:
        threshold = sorted(pred)[int(-sum(truth[truth==1]))]
        cur_pred = list(map(int, pred >= threshold))
    except Exception as e:
        print('sum', sum(truth[truth==1]))

    auroc = roc_auc_score(truth, pred)

    results = {'auroc': auroc}
    return results

def append_to_csv(d, csv_file):
    keys = d.keys()

    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            dictwriter = DictWriter(f, fieldnames=keys)
            dictwriter.writeheader()
            f.close()

    with open(csv_file, 'a') as f:
        dictwriter = DictWriter(f, fieldnames=keys)
        dictwriter.writerow(d)
        f.close()

def get_correlation(train_hard, train_err):
    from scipy.stats import rankdata, iqr, trim_mean, pearsonr, spearmanr

    coff, _ = pearsonr(train_hard.flatten(), train_err.flatten())
    spear, _ = spearmanr(train_hard.flatten(), train_err.flatten())

    return coff, spear


def get_eval_res_only_prior(train_hard, test_hard, train_err, test_err, test_y):
    res = {}
    
    # base
    res['base'] = eval_scores(test_err, test_y)

    # using prior
    res['only_hardness'] = eval_scores(test_hard, test_y)

    # prior z-score
    hard_prior = (test_hard - train_hard.mean())**2 / np.std(train_hard)
    res['only_prior'] = eval_scores(hard_prior, test_y)

    # print('res', res)
    return res

THRESHOLDS = ['median', 'q3']
PERS = [0.25, 0.1, 0.05, 0.01]
LAMBDAS = [0.01, 0.05, 0.1, 0.5, 1, 2, 5]


def get_eval_res(train_hard, test_hard, train_err, test_err, test_y, config = {}):
    import time
    import statsmodels.formula.api as smf

    res = {}
    scores = {}
    
    # base
    res['base'] = eval_scores(test_err, test_y)

    thres = THRESHOLDS
    percentiles = PERS
    lambdas = LAMBDAS
    
    if 'thres' in config:
        thres = config['thres']
    if 'percentile' in config:
        percentiles = config['percentile']
    if 'lambda' in config:
        lambdas = config['lambda']
    
    for thre in thres:
        for per in percentiles:

            start_time = time.time()

            qrs_model = smf.quantreg('y ~ x', dict(x=train_hard, y=train_err))

            q1_fit = qrs_model.fit(q=per)
            q2_fit = qrs_model.fit(q=0.5)
            q3_fit = qrs_model.fit(q=1-per)

            fit_time = time.time() - start_time

            q1 = np.asarray(q1_fit.predict(exog=dict(x=test_hard))).flatten()
            median = np.asarray(q2_fit.predict(exog=dict(x=test_hard))).flatten()
            q3 = np.asarray(q3_fit.predict(exog=dict(x=test_hard))).flatten()
            
            if thre == 'median':
                sc = np.abs(test_err - median) / (np.abs(q3 - q1) + 1e-2)
            elif thre == 'q3':
                sc = np.abs(test_err - q3) / (np.abs(q3 - q1) + 1e-2)


            predict_runtime = time.time() - start_time - fit_time
            cal_runtime = time.time() - start_time
            # print('cal_time', cal_runtime)
            # print('run time:', cal_runtime)

            train_q1 = np.asarray(q1_fit.predict(exog=dict(x=train_hard))).flatten()
            train_median = np.asarray(q2_fit.predict(exog=dict(x=train_hard))).flatten()
            train_q3 = np.asarray(q3_fit.predict(exog=dict(x=train_hard))).flatten()

            sc_train = (train_err - train_median) / (np.abs(train_q3 - train_q1) + 1e-2) 


            res[f'[{thre}]quantile_{per}'] = {**eval_scores(sc, test_y)}
            scores[f'[{thre}]quantile_{per}'] = deepcopy(sc)

            # add hardness prior
            orig_sc = sc
            for prior_lambda in lambdas:
                start = time.time()
                hard_prior = (test_hard - train_hard.mean())**2 / np.std(train_hard)
                new_sc = orig_sc + prior_lambda * hard_prior
                end = time.time() - start

                res[f'[{thre}]quantile_{per}_w/piror_{prior_lambda}'] = {**eval_scores(new_sc, test_y)}
                scores[f'[{thre}]quantile_{per}_w/piror_{prior_lambda}'] = deepcopy(new_sc)
                

    res['out'] = {'auroc': 0}
    scores['out'] = None
    for key, item in res.items():
        if 'quantile' in key and res['out']['auroc'] < res[key]['auroc']:
            res['out'] = dict({**item})
            scores['out'] = scores[key]

    return res, scores


def get_eval_res_wo_prior(train_hard, test_hard, train_err, test_err, test_y, config={}):
    import time
    import statsmodels.formula.api as smf

    res = {}
    scores = {}

    # base
    res['base'] = eval_scores(test_err, test_y)

    thres = THRESHOLDS
    percentiles = PERS
    lambdas = LAMBDAS

    if 'thres' in config:
        thres = config['thres']
    if 'percentile' in config:
        percentiles = config['percentile']
    if 'lambda' in config:
        lambdas = config['lambda']
    

    for thre in thres:
    # quantile regression
        for per in percentiles:

            start_time = time.time()

            qrs_model = smf.quantreg('y ~ x', dict(x=train_hard, y=train_err))

            q1_fit = qrs_model.fit(q=per)
            q2_fit = qrs_model.fit(q=0.5)
            q3_fit = qrs_model.fit(q=1-per)

            fit_time = time.time() - start_time

            q1 = np.asarray(q1_fit.predict(exog=dict(x=test_hard))).flatten()
            median = np.asarray(q2_fit.predict(exog=dict(x=test_hard))).flatten()
            q3 = np.asarray(q3_fit.predict(exog=dict(x=test_hard))).flatten()
            
            if thre == 'median':
                sc = np.abs(test_err - median) / (np.abs(q3 - q1) + 1e-2)
            elif thre == 'q3':
                sc = np.abs(test_err - q3) / (np.abs(q3 - q1) + 1e-2)


            predict_runtime = time.time() - start_time - fit_time
            cal_runtime = time.time() - start_time
            # print('cal_time', cal_runtime)
            # print('run time:', cal_runtime)

            train_q1 = np.asarray(q1_fit.predict(exog=dict(x=train_hard))).flatten()
            train_median = np.asarray(q2_fit.predict(exog=dict(x=train_hard))).flatten()
            train_q3 = np.asarray(q3_fit.predict(exog=dict(x=train_hard))).flatten()

            sc_train = (train_err - train_median) / (np.abs(train_q3 - train_q1) + 1e-2) 

            res[f'[{thre}]quantile_{per}'] = {**eval_scores(sc, test_y)}
            scores[f'[{thre}]quantile_{per}'] = deepcopy(sc)

    res['out'] = {'auroc': 0}
    for key, item in res.items():
        if 'quantile' in key and res['out']['auroc'] < res[key]['auroc']:
            res['out'] = dict({**item})
            scores['out'] = scores[key]

    return res, scores