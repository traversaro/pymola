import os
from collections import OrderedDict

import casadi as ca
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


class HybridDae:

    def __init__(self, **kwargs):
        self.x = OrderedDict()  # states (have derivatives)
        self.m = OrderedDict()  # discrete states
        self.pre_m = OrderedDict()  # discrete pre states
        self.p = OrderedDict()  # parameters and constants
        self.y = OrderedDict()  # algebraic states
        self.c = OrderedDict()  # conditions
        self.f_x = [] # continuous integration
        self.f_m = [] # discrete update
        self.f_c = [] # conditions
        self.g = [] # algebraic equations
        self.properties = {}

        # handle user args
        for k in kwargs.keys():
            if k in self.__dict__.keys():
                setattr(self, k, kwargs[k])
            else:
                raise ValueError('unknown argument', k)

    def __repr__(self):
        s = "\n"
        for x in ['x', 'm', 'pre_m', 'y', 'c', 'f_c', 'f_m', 'f_x']:
            v = getattr(self, x)
            if isinstance(v, OrderedDict):
                v = [str(v[e]) for e in v.keys()]
            elif isinstance(v, list):
                v = [str(e) for e in v]
            s += "{:6s}({:3d})\t:\t{:s}\n".format(x, len(v), str(v))
        return s


# noinspection PyProtectedMember
class ModelListener:
    """ Converts ModelicaXML file to Hybrid DAE"""

    def __init__(self):
        self.depth = 0
        self.model = {}
        self.scope_stack = []

        # Define an operator map that can be used as
        # self.opmap[n_operations][operator](*args)
        self.opmap = {
            1: {
                'der': self.der,
                '-': lambda x: -1 * x
            },
            2: {
            },
        }

    @property
    def scope(self):
        return self.scope_stack[-1]

    def call(self, tag_name: str, *args, **kwargs):
        """Convenience method for calling methods with walker."""
        if hasattr(self, tag_name):
            getattr(self, tag_name)(*args, **kwargs)

    def new_var(self, name, shape):
        """Create a new symbolic variable in scope."""
        return ca.SX.sym(name, *shape)

    def der(self, x: ca.SX):
        """Get the derivative of the variable, create it if it doesn't exist."""
        name = 'der({:s})'.format(x.name())
        if name not in self.scope['dvar'].keys():
            self.scope['dvar'][name] = ca.SX.sym(name, *x.shape)
        return self.scope['dvar'][name]

    def get_var(self, name):
        """Get the variable in the current scope"""
        return self.scope['var'][name]

    def log(self, *args, **kwargs):
        """Convenience function for printing indenting debug output."""
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

        # print model
        if self.model[tree] is not None:
            self.log('\033[1m', 'model:', self.model[tree], '\033[0m')

        # print name to log
        self.log('}', tree.tag)

    def enter_classDefinition(self, tree: etree._Element):
        # we don't know if variables are states
        # yet, we need to wait until equations are parsed
        self.scope_stack.append({
            'var': OrderedDict(), # variables
            'dvar': OrderedDict(), # derivative of variables
            'eqs': [], # equations
            'c': OrderedDict, # conditions
            'p': OrderedDict, # parameters and constants
            'properties': {}, # properties for variables
        })

    def exit_classDefinition(self, tree: etree._Element):
        class_scope = self.scope_stack.pop()
        dae = HybridDae()
        self.model[tree] = dae
        for v_name, v in class_scope['var'].items():
            if v_name in class_scope['dvar'].keys():
                dae.x[v_name] = v
            else:
                dae.x[v_name] = v
        dae.properties.update(class_scope['properties'])
        dae.f_x.extend(class_scope['eqs'])

    #-------------------------------------------------------------------------
    # component
    #-------------------------------------------------------------------------
    def enter_component(self, tree: etree._Element):
        self.scope_stack.append({
            'start': None,
            'fixed': None,
        })

    def exit_component(self, tree: etree._Element):
        var_scope = self.scope_stack.pop()
        name = tree.attrib['name']
        shape = (1, 1)
        sym = self.new_var(name, shape)
        self.model[tree] = sym
        self.scope['properties'][name] = var_scope
        self.scope['var'][name] = self.model[tree]

    def exit_local(self, tree: etree._Element):
        name = tree.attrib['name']
        self.model[tree] = self.get_var(name)

    def exit_operator(self, tree: etree._Element):
        op = tree.attrib['name']
        self.model[tree] = self.opmap[len(tree)][op](*[self.model[e] for e in tree])

    def exit_apply(self, tree: etree._Element):
        op = tree.attrib['builtin']
        self.model[tree] = self.opmap[len(tree)][op](*[self.model[e] for e in tree])

    def exit_equal(self, tree: etree._Element):
        self.model[tree] = self.model[tree[0]] - self.model[tree[1]]
        self.scope['eqs'].append(self.model[tree])

    def exit_equation(self, tree: etree._Element):
        # must be an equal equation since it is flattened
        assert len(tree) == 1
        self.model[tree] = self.model[tree[0]]

    def exit_item(self, tree: etree._Element):
        self.scope[tree.attrib['name']] = 1

    def exit_real(self, tree: etree._Element):
        self.model[tree] = ca.DM(float(tree.attrib["value"]))

    def enter_modelica(self, tree):
        pass

    def exit_true(self, tree):
        self.model[tree] = True


# noinspection PyProtectedMember
def walk(e: etree._Element, l: ModelListener):
    tag = e.tag
    l.call('enter_every_before', e)
    l.call('enter_' + tag, e)
    l.call('enter_every_after', e)
    for c in e.getchildren():
        walk(c, listener)
    l.call('exit_every_before', e)
    l.call('exit_' + tag, e)
    l.call('exit_every_after', e)


if __name__ == "__main__":
    parser = XMLParser(SCHEMA_DIR, 'Modelica.xsd')
    example_file = os.path.join(
        SCHEMA_DIR, 'examples', 'toplevel-example.xml')
    root = parser.read_file(example_file)
    listener = ModelListener()
    walk(root, listener)
