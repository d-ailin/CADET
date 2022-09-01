import pickle
import numpy as np
from numpy.lib.function_base import append
from csv import DictWriter
import os

from utils.eval_util import *
from utils.data_util import *

def main(tag, dataname, model_type):
    with open('./saved_error/{}_all/{}/{}_0.pkl'.format(tag, dataname, model_type), 'rb') as f:
        res = pickle.load(f)

    train_pred = res['train_pred']
    test_pred = res['test_pred']
    test_y = res['test_y']
    train_x = res['train_x']
    test_x = res['test_x']

    train_hard = get_hardness(train_x).squeeze(1)
    test_hard = get_hardness(test_x).squeeze(1)


    if 'test_err' in res:
        test_error = res['test_err']
        train_error = res['train_err']
    else:
        train_error = (np.abs(train_pred - train_x)).mean((1,2,3))
        test_error = (np.abs(test_pred - test_x)).mean((1,2,3))

    res, _ = get_eval_res(train_hard, test_hard, train_error, test_error, test_y)


    corr = get_correlation(train_hard, train_error)
    print(dataname, model_type)
    print('correlation (Pearson): ', corr[0])


    print('result:')
    print('Base AUROC:', res['base'])
    print('After AUROC', res['out'])

    # return res

import sys
if __name__ == '__main__':

    tag = sys.argv[1]
    dataname = sys.argv[2]
    model_type = sys.argv[3]

    main(tag, dataname, model_type)
