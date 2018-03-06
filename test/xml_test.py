#!/usr/bin/env python
"""
"""
import os
import sys
import time
import unittest

from pymola.backends.xml import hybrid_dae, modelica_xml_parser

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

    def test_xml(self):
        parser = modelica_xml_parser.XMLParser(
            modelica_xml_parser.SCHEMA_DIR, 'Modelica.xsd')
        example_file = os.path.join(
            modelica_xml_parser.FILE_PATH, 'bouncing-ball.xml')
        root = parser.read_file(example_file)
        listener = modelica_xml_parser.ModelListener(verbose=True)
        modelica_xml_parser.walk(root, listener)
        model = listener.model[root][0]  # type: HybridDae
        print(model)
        self.flush()
