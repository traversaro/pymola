import casadi as ca
import numpy as np

def new_mx(name, *shape):
    if len(shape) == 1 and not np.isscalar(shape[0]):
        shape = shape[0]

    obj = ca.MX.sym(name, *shape)
    obj._modelica_shape = tuple(shape)
    return obj


class MTensor:
    def __init__(self, name, *shape):
        self._shape = tuple(shape)
        self._mx = ca.MX.sym(name, np.prod(shape))

    @property
    def shape(self):
        return self._shape

    def __getattr__(self, attr):
        return getattr(self._mx, attr)

    def __getitem__(self, k):
        assert len(k) == len(self._shape)
        return self._mx[np.ravel_multi_index(tuple(k,), self._shape)]

