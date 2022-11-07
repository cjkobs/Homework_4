# here goes nothing

import numpy as np
import math as m

from sympy import symbols


def grg():
    x1, x2 = symbols('x1 x2 x3')
    xvars = [x1, x2]

    f = x1 ^ 2 + x2 ^ 2 + (x1 + x2) ^ 2
    h = (x1 ^ 2) / 4 + (x2 ^ 2) / 5 + ((x1 + x2) ^ 2) / 25 - 1
    alpha = 1.0
    b = 0.5
    t = 0.3
    max_iter = 100
    eps = 0.001

    x_init = np.array([0, 5 / m.sqrt(6)])

    df = np.array([np.diff(f, xvar) for xvar in xvars])
    dh = np.array([[np.diff(h, xvar) for xvar in xvars] for h in h])
    nonbasic_vars = len(xvars) - len(h)
    opt_sols = []

    for iter