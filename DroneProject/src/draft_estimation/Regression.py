import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def fit_and_predict_LMedS(x, y, line):
    pass


def fit_and_predict_LMS(x, y, line):
    n, m_x, m_y = np.size(x), np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_0, b_1 = m_y - b_1*m_x, SS_xy / SS_xx
    return np.array([b_0 + b_1*x for x in line])
  

def fit_and_predict_RANSAC(xs, ys, line):
    xs, ys, line = xs.reshape(-1, 1), ys.reshape(-1, 1), line.reshape(-1, 1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(xs, ys)
    return ransac.predict(line)
