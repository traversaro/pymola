from typing import Dict

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from .hybrid_dae import HybridOde


# noinspection PyCallByClass
def sim(model: HybridOde, options: Dict = None) -> Dict[str, np.array]:
    """
    Simulates a Dae model.
    """
    if model.f_x_rhs.shape[0] < 1:
        raise ValueError("there are no ODE equations to simulate, "
                         "check that the model is explicit")

    x0 = []
    for x in ca.vertsplit(model.x):
        start = model.prop[x.name()]['start']
        if start is None:
            Warning("using default start value for", x.name())
            x0.append(ca.DM.zeros(x.numel(), 1))
        else:
            x0.append(ca.reshape(start, x.numel(), 1))
    x0 = np.array(x0, dtype=float)

    p0 = []
    for x in ca.vertsplit(model.p):
        value = model.prop[x.name()]['value']
        if value is None:
            RuntimeError("no value for (param/constant)", x.name())
            p0.append(np.zeros(x.numel(), 1))
        else:
            p0.append(np.reshape(np.array(value, dtype=np.float), x.numel(), 1))
    p0 = np.array(p0, dtype=float)

    # set options
    opt = {
        'x0': x0,
        'p': p0,
        't0': 0,
        'tf': 1,
        'dt': 0.1,
        'integrator': 'vode',
    }
    if options is not None:
        for k in options.keys():
            if k in opt.keys():
                opt[k] = options[k]
            else:
                raise ValueError("unknown option {:s}".format(k))

    # create functions
    f_y = model.create_function_f_y()
    f_c = model.create_function_f_c()
    f_ode = model.create_function_f_x()
    f_J = model.create_function_f_J()

    # initialize sim loop
    t0 = opt['t0']
    tf = opt['tf']
    x = opt['x0']
    p = opt['p']
    dt = opt['dt']
    n = int(tf / dt)
    data = {
        't': np.arange(t0, tf, dt),
        'x': np.zeros((n, model.x.numel())),
        'y': np.zeros((n, model.y.numel())),
        'c': np.zeros((n, model.c.numel()))
    }

    # create integrator
    integrator = scipy.integrate.ode(f_ode, f_J)
    integrator.set_integrator(opt['integrator'])

    # run sim loop
    for i in range(0, n):
        t = t0 + dt * i
        c = np.array(f_c(t, x, p))
        p_vect = np.vstack([p, c])

        # setup integration steps
        integrator.set_f_params(p_vect)
        integrator.set_jac_params(p_vect)
        integrator.set_initial_value(x, integrator.t)

        # get x before step
        x_pre = integrator.y

        # perform integration step
        integrator.integrate(t)
        x = integrator.y

        # hard  coded logic for bouncing, ball, need to implement this
        if x[0] < 0 and not(x_pre[0] < 0):
            x[0] = 0
            x[1] = -0.7*x[1]
        elif (x[0] < 0 and abs(x[1]) < 0.01) and not (x_pre[0] < 0 and abs(x_pre[1]) < 0.01):
            x[0] = 0
            x[1] = 0

        # compute output
        y = f_y(x, p, t)

        # store data
        data['x'][i, :] = np.array(x)
        data['y'][i, :] = ca.vertsplit(y)
        data['c'][i, :] = ca.vertsplit(c)

    data['labels'] = {}
    for f in ['x', 'y', 'c']:
        data['labels'][f] = [x.name() for x in ca.vertsplit(getattr(model, f))]
    return data


def plot(data, fields=None):
    if fields is None:
        fields = ['x', 'y', 'c']
    labels = []
    lines = []
    for f in fields:
        if min(data[f].shape) > 0:
            f_lines = plt.plot(data['t'], data[f], '-', alpha=0.5)
            lines.extend(f_lines)
            labels.extend(data['labels'][f])
    plt.legend(lines, labels)
    plt.xlabel('t, sec')
    plt.grid()
