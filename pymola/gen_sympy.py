from __future__ import print_function, absolute_import, division, print_function, unicode_literals
from . import tree

import jinja2
import os
import sys
import copy

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class SympyGenerator(tree.TreeListener):

    def __init__(self):
        super(SympyGenerator, self).__init__()
        self.src = {}

    def exitFile(self, tree):
        d = {'classes': []}
        for key in sorted(tree.classes.keys()):
            d['classes'] += [self.src[tree.classes[key]]]

        template = jinja2.Template('''
# do not edit, generated by pymola

from __future__ import print_function, division
import sympy
import sympy.physics.mechanics as mech
from pymola.sympy_runtime import OdeModel
from sympy import sin, cos, tan
{%- for class_key, class in tree.classes.items() %}
{{ render.src[class] }}
{%- endfor %}
''')
        self.src[tree] = template.render({
            'tree': tree,
            'render': self,
        })

    def exitClass(self, tree):
        states = []
        inputs = []
        outputs = []
        constants = []
        parameters = []
        variables = []
        symbols = sorted(tree.symbols.values(), key=lambda s: s.order)
        for s in symbols:
            if len(s.prefixes) == 0:
                variables += [s]
            else:
                for prefix in s.prefixes:
                    if prefix == 'state':
                        states += [s]
                    elif prefix == 'constant':
                        constants += [s]
                    elif prefix == 'parameter':
                        parameters += [s]
                    elif prefix == 'input':
                        inputs += [s]
                    elif prefix == 'output':
                        outputs += [s]

        for s in outputs:
            if s not in states:
                variables += [s]

        states_str = ', '.join([self.src[s] for s in states])
        inputs_str = ', '.join([self.src[s] for s in inputs])
        outputs_str = ', '.join([self.src[s] for s in outputs])
        constants_str = ', '.join([self.src[s] for s in constants])
        parameters_str = ', '.join([self.src[s] for s in parameters])
        variables_str = ', '.join([self.src[s] for s in variables])

        d = locals()
        d.pop('self')
        d['render'] = self

        template = jinja2.Template('''

class {{tree.name}}(OdeModel):

    def __init__(self):

        super({{tree.name}}, self).__init__()

        # states
        {% if states_str|length > 0 -%}
        {{ states_str }} = mech.dynamicsymbols('{{ states_str|replace('__', '.') }}')
        {% endif -%}
        self.x = sympy.Matrix([{{ states_str }}])
        self.x0 = {
            {% for s in states -%}
            {{render.src[s]}} : {{tree.symbols[s.name].start.value}},
            {% endfor -%}}

        # variables
        {% if variables_str|length > 0 -%}
        {{ variables_str }} = mech.dynamicsymbols('{{ variables_str|replace('__', '.') }}')
        {% endif -%}
        self.v = sympy.Matrix([{{ variables_str }}])

        # constants
        {% if constants_str|length > 0 -%}
        {{ constants_str }} = sympy.symbols('{{ constants_str|replace('__', '.') }}')
        {% endif -%}
        self.c = sympy.Matrix([{{ constants_str }}])
        self.c0 = {
            {% for s in constants -%}
            {{render.src[s]}} : {{tree.symbols[s.name].start.value}},
            {% endfor -%}}

        # parameters
        {% if parameters_str|length > 0 -%}
        {{ parameters_str }} = sympy.symbols('{{ parameters_str|replace('__', '.') }}')
        {% endif -%}
        self.p = sympy.Matrix([{{ parameters_str }}])
        self.p0 = {
            {% for s in parameters -%}
            {{render.src[s]}} : {{tree.symbols[s.name].start.value}},
            {% endfor -%}}

        # inputs
        {% if inputs_str|length > 0 -%}
        {{ inputs_str }} = mech.dynamicsymbols('{{ inputs_str|replace('__', '.') }}')
        {% endif -%}
        self.u = sympy.Matrix([{{ inputs_str }}])
        self.u0 = {
            {% for s in inputs -%}
            {{render.src[s]}} : {{tree.symbols[s.name].start.value}},
            {% endfor -%}}

        # outputs
        {% if outputs_str|length > 0 -%}
        {{ outputs_str }} = mech.dynamicsymbols('{{ outputs_str|replace('__', '.') }}')
        {% endif -%}
        self.y = sympy.Matrix([{{ outputs_str }}])

        # equations
        self.eqs = [
            {% for eq in tree.equations -%}
            {{ render.src[eq] }},
            {% endfor -%}
        ]

        self.compute_fg()
''')
        self.src[tree] = template.render(d)

    def exitExpression(self, tree):
        op = str(tree.operator)
        n_operands = len(tree.operands)
        if op == 'der':
            src = '({var:s}).diff(self.t)'.format(
                var=self.src[tree.operands[0]])
        elif op in ['*', '+', '-', '/'] and n_operands == 2:
            src = '{left:s} {op:s} {right:s}'.format(
                op=op,
                left=self.src[tree.operands[0]],
                right=self.src[tree.operands[1]])
        elif op in ['+', '-'] and n_operands == 1:
            src = '{op:s} {expr:s}'.format(
                op=op,
                expr=self.src[tree.operands[0]])
        else:
            src = "({operator:s} ".format(**tree.__dict__)
            for operand in tree.operands:
                src +=  ' ' + self.src[operand]
            src += ")"
        self.src[tree] = src

    def exitPrimary(self, tree):
        val = str(tree.value)
        self.src[tree] = "{:s}".format(val)

    def exitComponentRef(self, tree):
        self.src[tree] = "{name:s}".format(name=tree.name.replace('.','__'))

    def exitSymbol(self, tree):
        self.src[tree] = "{name:s}".format(name=tree.name.replace('.','__'))

    def exitEquation(self, tree):
        self.src[tree] = "{left:s} - ({right:s})".format(
            left=self.src[tree.left],
            right=self.src[tree.right])

    def exitConnectClause(self, tree):
        #print('class context', type(self.context['Class'].symbols[tree.left.name]))
        self.src[tree] = "{left:s} - ({right:s})".format(
            left=self.src[tree.left],
            right=self.src[tree.right])


def generate(ast_tree, model_name):
    ast_tree_new = copy.deepcopy(ast_tree)
    ast_walker = tree.TreeWalker()
    flat_tree = tree.flatten(ast_tree_new, model_name)
    sympy_gen = SympyGenerator()
    ast_walker.walk(sympy_gen, flat_tree)
    return sympy_gen.src[flat_tree]
