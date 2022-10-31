import numpy as np
ALPHA = 0.8


def net_fun(w, x):
    '''Dot product between a weight row and input vector'''
    return np.dot(w, x)


def act_tanh(x, alpha=ALPHA):
    return np.tanh(alpha * x / 2)


def act_ltu(x):
    return np.heaviside(x, 0)

def act_sigmoid(x, alpha=ALPHA):
    return 1/(1+np.exp(-alpha*x))


def derivative_tanh(x, alpha=ALPHA):
    '''derivative of tanh act func'''
    return (1 - (np.tanh(alpha * x / 2)) ** 2)