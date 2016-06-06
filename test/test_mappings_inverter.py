#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__author__ = "Yuji Ikeda"

import unittest
import numpy as np
from fc_analysis.fc_analyzer import MappingsInverter


class TestMappingsInverter(unittest.TestCase):
    def test_0(self):
        mappings = [
            [0, 1, 2, 3],
        ]
        mappings_inverse_obtained = (
            MappingsInverter().invert_mappings(mappings)
        )
        mappings_inverse_expected = [
            [0, 1, 2, 3],
        ]
        self.assertTrue(
            np.alltrue(mappings_inverse_obtained == mappings_inverse_expected)
        )


    def test_1(self):
        mappings = [
            [1, 2, 0, 3],
        ]
        mappings_inverse_obtained = (
            MappingsInverter().invert_mappings(mappings)
        )
        mappings_inverse_expected = [
            [2, 0, 1, 3],
        ]
        self.assertTrue(
            np.alltrue(mappings_inverse_obtained == mappings_inverse_expected)
        )


if __name__ == "__main__":
    unittest.main()
