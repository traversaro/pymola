import casadi as ca
from typing import List, Dict, Union, Any
import numpy as np
import scipy.integrate

Sym = Union[ca.SX, ca.MX]


def split_dae_alg(eqs: Sym, dx: Sym) -> Dict[str, Sym]:
    dae = []
    alg = []
    for eq in ca.vertsplit(eqs):
        if ca.depends_on(eq, dx):
            dae.append(eq)
        else:
            alg.append(eq)
    return {
        'dae': ca.vertcat(*dae),
        'alg': ca.vertcat(*alg)
    }


def permute(x: Sym, perm: List[int]) -> Sym:
    x_s = []
    for i in perm:
        x_s.append(x[i])
    return ca.vertcat(*x_s)


# noinspection PyPep8Naming,SpellCheckingInspection
def blt(f: List[Sym], x: List[Sym]) -> Dict[str, Any]:
    """
    Sort equations by dependence
    """
    J = ca.jacobian(f, x)
    nblock, rowperm, colperm, rowblock, colblock, coarserow, coarsecol = J.sparsity().btf()
    return {
        'J': J,
        'nblock': nblock,
        'rowperm': rowperm,
        'colperm': colperm,
        'rowblock': rowblock,
        'colblock': colblock,
        'coarserow': coarserow,
        'coarsecol': coarsecol
    }


# noinspection PyPep8Naming
def blt_linear_approx(f: Sym, x: Sym, a: Sym=None, assert_linear: bool=False) -> Dict[str, Sym]:
    """
    0 = f(x) = f(a) + J*x   # taylor series about 0 (if f(x) linear in x, then globally valid)
    J*x = -f(a)             # solve for x
    where J = df/dx
    """
    if a is None:
        a = ca.DM.zeros(x.numel(), 1)

    # sort
    res_blt = blt(f, x)
    if assert_linear and ca.depends_on(res_blt['J'], x):
        raise RuntimeError('not linear')
    f_s = permute(f, res_blt['rowperm'])
    x_s = permute(x, res_blt['colperm'])

    # solve
    rhs = []
    x_rhs = []
    for i in range(res_blt['nblock']):
        rows = res_blt['rowblock']
        cols = res_blt['rowblock']
        f_b = f_s[rows[i]:rows[i + 1]]
        x_b = x_s[cols[i]:cols[i + 1]]
        J_b = ca.jacobian(f_b, x_b)
        J_rank = ca.sprank(J_b)
        assert J_rank == x_b.shape[0]
        f_res = ca.substitute(f_b, x, a)
        rhs_b = ca.solve(J_b, -f_res)
        x_rhs.extend(ca.vertsplit(x_b))
        rhs.append(rhs_b)
    rhs = ca.vertcat(*rhs)
    x_rhs = ca.vertcat(*x_rhs)

    # unpermute
    rhs = permute(rhs, res_blt['colperm'])
    x_rhs = permute(x_rhs, res_blt['colperm'])

    # make sure we define column vectors
    rhs = ca.reshape(rhs, rhs.numel(), 1)
    x_rhs = ca.reshape(x_rhs, rhs.numel(), 1)

    return {
        'rhs': rhs,
        'x_rhs': x_rhs
    }


# noinspection PyPep8Naming
class HybridOde:

    def __init__(self, sym: type=ca.SX, **kwargs):
        self.c = sym(0, 1)  # conditions
        self.dx = sym(0, 1)  # states derivatives
        self.f_c = sym(0, 1)  # condition relations
        self.f_m = sym(0, 1)  # discrete update
        self.f_x_rhs = sym(0, 1)  # continuous integration
        self.g_rhs = sym(0, 1)  # algebraic states as a function of state
        self.m = sym(0, 1)  # discrete states
        self.p = sym(0, 1)  # parameters and constants
        self.pre_m = sym(0, 1)  # discrete pre states
        self.prop = {}  # properties
        self.sym = sym  # symbol type
        self.t = sym()  # time
        self.x = sym(0, 1)  # states (have derivatives)

        # handle user args
        for k in kwargs.keys():
            if k in self.__dict__.keys():
                setattr(self, k, kwargs[k])
            else:
                raise ValueError('unknown argument', k)

    def __repr__(self):
        s = "\n"
        for x in ['c', 'dx', 'f_c', 'f_m', 'f_x_rhs', 'g_rhs', 'm', 'p', 'pre_m', 'x']:
            v = getattr(self, x)
            s += "{:6s}({:3d}):\t{:s}\n".format(x, v.shape[0], str(v))
        return s


class HybridDae:

    def __init__(self, sym: type=ca.SX, **kwargs):
        self.sym = sym
        self.c = sym(0, 1)  # conditions
        self.dx = sym(0, 1)  # states derivatives
        self.f_c = sym(0, 1)  # condition relations
        self.f_m = sym(0, 1)  # discrete update
        self.f_x = sym(0, 1)  # continuous integration
        self.m = sym(0, 1)  # discrete states
        self.p = sym(0, 1)  # parameters and constants
        self.pre_m = sym(0, 1)  # discrete pre states
        self.prop = {}  # properties
        self.t = sym()  # time
        self.x = sym(0, 1)  # states (have derivatives)
        self.y = sym(0, 1)  # algebraic states

        # handle user args
        for k in kwargs.keys():
            if k in self.__dict__.keys():
                setattr(self, k, kwargs[k])
            else:
                raise ValueError('unknown argument', k)

    def __repr__(self):
        s = "\n"
        for x in ['c', 'dx', 'f_c', 'f_m', 'f_x', 'm', 'p', 'pre_m', 'x', 'y']:
            v = getattr(self, x)
            s += "{:6s}({:3d}):\t{:s}\n".format(x, v.shape[0], str(v))
        return s

    def to_ode(self) -> HybridOde:
        res_split = split_dae_alg(self.f_x, self.dx)
        alg = res_split['alg']
        dae = res_split['dae']

        res_ode = blt_linear_approx(dae, self.dx, assert_linear=True)
        res_g = blt_linear_approx(alg, self.y, assert_linear=True)

        return HybridOde(
            c=self.c,
            dx=self.dx,
            f_c=self.f_c,
            f_m=self.f_m,
            f_x_rhs=res_ode['rhs'],
            m=self.m,
            p=self.p,
            pre_m=self.pre_m,
            prop=self.prop,
            sym=self.sym,
            x=self.x,
            g_rhs=res_g['rhs'],
        )



