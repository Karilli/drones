import numpy as np

from sklearn import linear_model


def fit_and_predict_LMedS(x, y, line):
    pass # TODO: not implemented


def fit_LMS(xs, ys):
    n, m_x, m_y = np.size(xs), np.mean(xs), np.mean(ys)
    SS_xy = np.sum(ys*xs) - n*m_y*m_x
    SS_xx = np.sum(xs*xs) - n*m_x*m_x
    b_0, b_1 = m_y - b_1*m_x, SS_xy / SS_xx
    return b_0, b_1
  

def predict_LMS(regressor, xs):
    b_0, b_1 = regressor
    return np.array([b_0 + b_1*x for x in xs])


def fit_and_predict_LMS(xs, ys, line):
    return predict_LMS(fit_LMS(xs, ys), line)


def fit_RANSAC(xs, ys):
    xs, ys = xs.reshape(-1, 1), ys.reshape(-1, 1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(xs, ys)
    return ransac


def predict_RANSAC(regressor, xs):
    xs = xs.reshape(-1, 1)
    return regressor.predict(xs)


def fit_and_predict_RANSAC(xs, ys, line):
    return predict_RANSAC(fit_RANSAC(xs, ys), line)
