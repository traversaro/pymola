#!/usr/bin/env python
"""
Test XML backend
"""
import os
import sys
import time
import unittest

import casadi as ca
import matplotlib.pyplot as plt

from pymola.backends.xml import hybrid_dae, modelica_xml_parser
from pymola.backends.xml import sim_scipy

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(TEST_DIR, 'models')
GENERATED_DIR = os.path.join(TEST_DIR, 'generated')


class XmlTest(unittest.TestCase):
    """
    Xml tests
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def flush():
        sys.stdout.flush()
        sys.stdout.flush()
        time.sleep(0.01)

    def test_bouncing_ball(self):
        # parse
        example_file = os.path.join(
            modelica_xml_parser.FILE_PATH, 'bouncing-ball.xml')
        model = modelica_xml_parser.parse(example_file)

        # convert to ode
        model_ode = model.to_ode()  # type: hybrid_dae.HybridOde

        # simulate
        data = sim_scipy.sim(model_ode, {'tf': 10, 'dt': 0.01})

        # plot
        sim_scipy.plot(data)
        plt.show()
        self.flush()
