import json
import casadi as ca


class HybridDae:

    def __init__(self, sym=ca.SX, **kwargs):
        self.sym = sym
        self.x = sym(0, 1)  # states (have derivatives)
        self.dx = sym(0, 1)  # states derivatives
        self.m = sym(0, 1)  # discrete states
        self.pre_m = sym(0, 1)  # discrete pre states
        self.p = sym(0, 1)  # parameters and constants
        self.y = sym(0, 1)  # algebraic states
        self.f_x = sym(0, 1)  # continuous integration
        self.f_m = sym(0, 1)  # discrete update
        self.c = sym(0, 1)  # conditions
        self.f_c = sym(0, 1)  # condition relations
        self.g = sym(0, 1)  # algebraic equations
        self.properties = {}

        # handle user args
        for k in kwargs.keys():
            if k in self.__dict__.keys():
                setattr(self, k, kwargs[k])
            else:
                raise ValueError('unknown argument', k)

    def vec(self, name):
        return ca.vertcat(*getattr(self, name))

    def __repr__(self):
        s = "\n"
        for x in ['x', 'dx', 'm', 'pre_m', 'y', 'p', 'c', 'f_c', 'f_m', 'f_x']:
            v = getattr(self, x)
            s += "{:6s}({:3d}):\t{:s}\n".format(x, v.shape[0], str(v))
        s += "{:s}".format(json.dumps(self.properties, indent=2))
        return s
