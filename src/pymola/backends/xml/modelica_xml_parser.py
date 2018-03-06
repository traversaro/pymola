from .hybrid_dae import HybridDae
import os
from collections import OrderedDict
import typing

import casadi as ca
# noinspection PyPackageRequirements
from lxml import etree

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(FILE_PATH, 'ModelicaXML', 'schemas')

class XMLParser:

    def __init__(self, schema_dir, schema_file):
        orig_path = os.path.abspath(os.curdir)
        os.chdir(schema_dir)
        with open(schema_file, 'r') as f:
            schema = etree.XMLSchema(etree.XML(f.read().encode('utf-8')))
        os.chdir(orig_path)
        self._parser = etree.XMLParser(schema=schema)

    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            xml_file = f.read().encode('utf-8')
        return etree.fromstring(xml_file, self._parser)


# noinspection PyProtectedMember,PyPep8Naming
class ModelListener:
    """ Converts ModelicaXML file to Hybrid DAE"""

    def __init__(self, sym: type=ca.SX, verbose=False):
        self.depth = 0
        self.model = {}
        self.scope_stack = []
        self.verbose = verbose
        self.sym = sym

        # Define an operator map that can be used as
        # self.op_map[n_operations][operator](*args)
        self.op_map = {
            1: {
                'der': self.der,
                '-': lambda x: -1 * x,
            },
            2: {
                '+': lambda x, y: x + y,
                '-': lambda x, y: x - y,
                '*': lambda x, y: x * y,
                '/': lambda x, y: x / y,
                '^': lambda x, y: ca.power(x, y),
                '>': lambda x, y: x > y,
                'lt': lambda x, y: x < y,
                'lt=': lambda x, y: x <= y,
                'reinit': self.reinit,
            },
        }

    @property
    def scope(self):
        return self.scope_stack[-1]

    def call(self, tag_name: str, *args, **kwargs):
        """Convenience method for calling methods with walker."""
        if hasattr(self, tag_name):
            getattr(self, tag_name)(*args, **kwargs)

    def der(self, x: ca.SX):
        """Get the derivative of the variable, create it if it doesn't exist."""
        name = 'der({:s})'.format(x.name())
        if name not in self.scope['dvar'].keys():
            self.scope['dvar'][name] = self.sym.sym(name, *x.shape)
            self.scope['states'].append(x.name())
        return self.scope['dvar'][name]

    @staticmethod
    def reinit(x_old, x_new):
        return 'reinit', x_old, x_new

    @staticmethod
    def get_attr(e, name, default):
        if name in e.attrib.keys():
            return e.attrib[name]
        else:
            return default

    def get_var(self, name):
        """Get the variable in the current scope"""
        return self.scope['var'][name]

    def log(self, *args, **kwargs):
        """Convenience function for printing indenting debug output."""
        if self.verbose:
            print('   ' * self.depth, *args, **kwargs)

    def enter_every_before(self, tree: etree._Element):
        # initialize model to None
        self.model[tree] = None

        # print name to log
        self.log(tree.tag, '{')

        # increment depth
        self.depth += 1

    def exit_every_after(self, tree: etree._Element):
        # decrement depth
        self.depth -= 1

        # self.log('tree:', etree.tostring(tree))

        # print model
        if self.model[tree] is not None:
            self.log('model:', self.model[tree])

        # print name to log
        self.log('}', tree.tag)

    # noinspection PyUnusedLocal
    def enter_classDefinition(self, tree: etree._Element):
        # we don't know if variables are states
        # yet, we need to wait until equations are parsed
        self.scope_stack.append({
            'time': self.sym.sym('time'),
            'var': OrderedDict(),  # variables
            'states': [],  # list of which variables are states (based on der call)
            'dvar': OrderedDict(),  # derivative of variables
            'eqs': [],  # equations
            'when_eqs': [],  # when equations
            'c': {},  # conditions
            'p': [],  # parameters and constants
            'prop': {},  # properties for variables
        })

    def exit_classDefinition(self, tree: etree._Element):
        dae = HybridDae()
        self.model[tree] = dae
        for var_name, v in self.scope['var'].items():
            variability = self.scope['prop'][var_name]['variability']
            if variability == 'continuous':
                if var_name in self.scope['states']:
                    dae.x = ca.vertcat(dae.x, v)
                    dae.dx = ca.vertcat(dae.dx, self.der(v))
                else:
                    dae.y = ca.vertcat(dae.y, v)
            elif variability == 'discrete':
                dae.m = ca.vertcat(dae.m, v)
            elif variability == 'parameter':
                dae.p = ca.vertcat(dae.p, v)
            elif variability == 'constant':
                dae.p = ca.vertcat(dae.p, v)
            else:
                raise ValueError('unknown variability', variability)

        dae.prop.update(self.scope['prop'])
        c_dict = self.scope['c']
        dae.c = ca.vertcat(dae.c, ca.vertcat(*[k for k in c_dict]))
        dae.f_c = ca.vertcat(dae.f_c, ca.vertcat(*[ c_dict[k] for k in c_dict]))
        dae.f_x = ca.vertcat(dae.f_x, ca.vertcat(*self.scope['eqs']))
        dae.t = self.scope['time']
        # dae.f_x.extend(self.scope['when_eqs'])
        self.scope_stack.pop()

    def enter_component(self, tree: etree._Element):
        self.model[tree] = {
            'start': None,
            'fixed': None,
            'value': None,
            'variability': self.get_attr(tree, 'variability', 'continuous'),
            'visibility': self.get_attr(tree, 'visibility', 'public'),
        }
        self.scope_stack.append(self.model[tree])

    def exit_component(self, tree: etree._Element):
        var_scope = self.scope_stack.pop()
        name = tree.attrib['name']
        shape = (1, 1)
        sym = ca.SX.sym(name, *shape)
        self.scope['prop'][name] = var_scope
        self.scope['var'][name] = sym

    def exit_local(self, tree: etree._Element):
        name = tree.attrib['name']
        self.model[tree] = self.get_var(name)

    def exit_operator(self, tree: etree._Element):
        op = tree.attrib['name']
        self.model[tree] = self.op_map[len(tree)][op](*[self.model[e] for e in tree])

    def exit_apply(self, tree: etree._Element):
        op = tree.attrib['builtin']
        self.model[tree] = self.op_map[len(tree)][op](*[self.model[e] for e in tree])

    def exit_equal(self, tree: etree._Element):
        self.model[tree] = self.model[tree[0]] - self.model[tree[1]]
        self.scope['eqs'].append(self.model[tree])

    def exit_equation(self, tree: etree._Element):
        # must be an equal equation since it is flattened
        assert len(tree) == 1
        self.model[tree] = self.model[tree[0]]

    def exit_modifier(self, tree: etree._Element):
        props = {}
        for e in tree:
            props.update(self.model[e])
        self.model[tree] = props
        self.scope.update(props)

    def exit_item(self, tree: etree._Element):
        self.model[tree] = {
            tree.attrib['name']: self.model[tree[0]]
        }

    def exit_real(self, tree: etree._Element):
        self.model[tree] = float(tree.attrib["value"])

    def exit_true(self, tree: etree._Element):
        self.model[tree] = True

    def exit_false(self, tree: etree._Element):
        self.model[tree] = False

    def exit_modelica(self, tree: etree._Element):
        self.model[tree] = [self.model[e] for e in tree[0]]

    def exit_when(self, tree: etree._Element):
        c = ca.SX.sym('c_{:d}'.format(len(self.scope['c'])))
        cond = self.model[tree[0]]
        then = self.model[tree[1]]
        self.scope['c'][c] = cond
        self.model[tree] = {
            'cond': c,
            'then': then,
        }
        self.scope['when_eqs'].append(self.model[tree])

    def exit_cond(self, tree: etree._Element):
        self.model[tree] = self.model[tree[0]]

    def exit_then(self, tree: etree._Element):
        self.model[tree] = self.model[tree[0]]


# noinspection PyProtectedMember
def walk(e: etree._Element, l: ModelListener):
    tag = e.tag
    l.call('enter_every_before', e)
    l.call('enter_' + tag, e)
    l.call('enter_every_after', e)
    for c in e.getchildren():
        walk(c, l)
    l.call('exit_every_before', e)
    l.call('exit_' + tag, e)
    l.call('exit_every_after', e)


if __name__ == "__main__":
    parser = XMLParser(SCHEMA_DIR, 'Modelica.xsd')
    example_file = os.path.join(
        FILE_PATH, 'bouncing-ball.xml')
    root = parser.read_file(example_file)
    listener = ModelListener(verbose=True)
    walk(root, listener)
    model = listener.model[root][0]  # type: HybridDae
    print(model)
    # print(type(model.x[0]))
