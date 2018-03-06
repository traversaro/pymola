import casadi as ca
import scipy.integrate
from typing import Dict
import numpy as np

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
    p0 = []
    for x in ca.vertsplit(model.p):
        value = model.prop[x.name()]['value']
        if value is None:
            RuntimeError("no value for (param/constant)", x.name())
            p0.append(np.zeros(x.numel(), 1))
        else:
            p0.append(np.reshape(np.array(value, dtype=np.float), x.numel(), 1))

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

    # Use just-in-time compilation to speed up the evaluation
    if ca.Importer.has_plugin('clang'):
        with_jit = True
        compiler = 'clang'
    elif ca.Importer.has_plugin('shell'):
        with_jit = True
        compiler = 'shell'
    else:
        print("WARNING; running without jit. "
              "This may result in very slow evaluation times")
        with_jit = False
        compiler = ''
    func_opt = {'jit': with_jit, 'compiler': compiler}

    # create output function
    output_func = ca.Function(
        'y',
        [model.x, model.p, model.t],
        [model.g_rhs], func_opt)

    # initialize sim loop
    t0 = opt['t0']
    tf = opt['tf']
    x = opt['x0']
    p = opt['p']
    y = ca.vertsplit(output_func(ca.vertcat(*x), ca.vertcat(*p), t0))
    dt = opt['dt']
    n = int(tf / dt)
    data = {
        't': np.arange(t0, tf, dt),
        'x': np.zeros((n, len(x))),
        'y': np.zeros((n, len(y)))
    }
    data['t'][0] = t0
    data['x'][0, :] = x
    data['y'][0, :] = y

    # create integrator
    f_ode = ca.Function(
        'f',
        [model.t, model.x, model.p],
        [model.f_x_rhs], func_opt)
    f_J = ca.Function(
        'J',
        [model.t, model.x, model.p],
        [ca.jacobian(model.f_x_rhs, model.x)], func_opt)
    integrator = scipy.integrate.ode(f_ode, f_J)
    integrator.set_initial_value(x, t0)
    integrator.set_f_params(p)
    integrator.set_jac_params(p)
    integrator.set_integrator(opt['integrator'])

    # run sim loop
    for i in range(1, n):
        t = t0 + dt * i
        integrator.integrate(t)
        x = integrator.y

        # compute output (this takes awhile, need to see how to speed it up)
        # it could be skipped all together or only computer when the user
        # asks for the variables for plotting after the simulation, but this
        # prevents the user from passing a control based on the output
        # y = output_func(ca.vertcat(*x), ca.vertcat(*p), t)

        # store data
        data['x'][i, :] = np.array(x)
        data['y'][i, :] = np.array(y)

    return data

