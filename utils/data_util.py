import numpy as np

def adjust_proportions(x, y, anom_prop, data_prop, norm_classes, anom_classes):

    idx_norm = np.where(np.isin(y, norm_classes))[0]
    idx_anom = np.where(np.isin(y, anom_classes))[0]
    idx_sample = np.random.choice(np.arange(len(idx_anom)), 
                                  int(len(idx_anom)*anom_prop), replace=False)
    x = np.concatenate((x[idx_norm], x[idx_anom][idx_sample]))
    y_org = np.concatenate((y[idx_norm], y[idx_anom][idx_sample]))
    y = np.asarray(list(map(int, np.isin(y_org, anom_classes))))
    idx_sample = np.random.choice(np.arange(len(x)), 
                                  int(len(x)*data_prop), replace=False)
    x = x[idx_sample]
    y = y[idx_sample]
    y_org = y_org[idx_sample]
    return x, y, y_org, idx_sample

def compute_error(y, y_pred):
    n_dim = len(y.shape)
    return ((y - y_pred)**2).mean(axis=tuple(range(1, n_dim-1)))


def get_hardness(x):
    if x.shape[1] == 1:
        x = x.transpose((0, 2, 3, 1))
    n, m1, m2  = x.shape[:3]
    return (np.mean(np.abs(x[:, :(m1-1), :] - x[:, 1:, :]), axis=(1,2)) + np.mean(np.abs(x[:, :, :(m2-1)] - x[:, :, 1:]), axis=(1,2)))

