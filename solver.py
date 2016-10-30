import theano.tensor as T
import theano
import numpy as np

def R2loss(y_true, y_pred):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    tot = T.sum(T.sqr(T.sub(y_true, T.mean(y_true))))
    res = T.sum(T.sqr(T.sub(y_true, y_pred)))
    return T.true_div(res, tot)