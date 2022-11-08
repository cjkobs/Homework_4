# here goes nothing

import numpy as np
import math as m

from sympy import symbols


def grg():
    x1, x2 = symbols('x1 x2 x3')

    # x1 is s, x2 is d
    xvars = [x1, x2]

    fx = x1 ** 2 + x2 ** 2 + (x1 + x2) ** 2
    hxs = (x1 ** 2) / 4 + (x2 ** 2) / 5 + ((x1 + x2) ** 2) / 25 - 1
    alpha_0 = 1.0
    b = 0.5
    t = 0.3
    max_iter = 100
    eps = 0.001

    x_init = np.array([0, 5 / m.sqrt(6)])

    dfx = np.array([np.diff(fx, xvar) for xvar in xvars])
    dhxs = np.array([[np.diff(hx, xvar) for xvar in xvars] for hx in hxs])
    nonbasic_vars = len(xvars) - len(hxs)
    opt_sols = []

    for outer_iter in range(max_iter):

        print '\n\nOuter loop iteration: {0}, optimal solution: {1}'.format(outer_iter + 1, x_init)
        opt_sols.append(fx.subs(zip(xvars, x_init)))

        d_f = np.array([df.subs(zip(xvars, x_init)) for df in dfx])
        d_h = np.array([[dh.subs(zip(xvars, x_init)) for dh in dhx] for dhx in dhxs])
        dh_ds = np.array([dhx[nonbasic_vars:] for dhx in d_h])
        dh_dd = np.array([dhx[:nonbasic_vars] for dhx in d_h])
        df_ds = d_f[nonbasic_vars:]
        df_dd = d_f[:nonbasic_vars]

        dh_ds_inv = np.linalg.inv(np.array(dh_ds))

        dfdd = df_dd - np.matmul(np.matmul(df_ds, dh_ds_inv), dh_dd)

        if (dfdd[0]) ** 2 <= eps:
            break
        # right hand side of f(alpha) equation
        rhs = np.matmul(dh_ds_inv, np.matmul(dh_dd, dfdd.T)).T

        alpha = alpha_0

        while alpha > 0.001:

            f_alpha = x_init.T + alpha * rhs
            f_alpha_bar = f_alpha[:nonbasic_vars]
            f_alpha_cap = f_alpha[nonbasic_vars:]
            flag = False

            for iter in range(max_iter):
                print 'Iteration: {0}, optimal solution obtained at x = {1}'.format(iter + 1, f_alpha)
                h = np.array([hx.subs(zip(xvars, f_alpha)) for hx in hxs])
                if all([h_i ** 2 < eps for h_i in h]):
                    if fx.subs(zip(xvars, x_init)) <= fx.subs(zip(xvars, f_alpha)):
                        alpha = alpha * b
                        break
                    else:
                        x_init = f_alpha
                        flag = True
                        break

                df_d = np.array([[dh.subs(zip(xvars, f_alpha)) for dh in dhx] for dhx in dhxs])
                d_h_s = np.linalg.inv(np.array([dhx[nonbasic_vars:] for dhx in d_h_d], dtype=float))
                d_k = f_alpha_cap - np.matmul(d_h_s, h)



