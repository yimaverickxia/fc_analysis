#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO(ikeda): The function "get_rotations_cart" is not appropriate for this
#     module.
# TODO(ikeda): The structure of the variable "force_constants_pair" should be
#     modified. We want to use numpy functions.
from __future__ import absolute_import, division, print_function
import numpy as np
from phonopy.structure import spglib


class MappingsInverter(object):
    def invert_mappings(self, mappings):
        mappings_inverse = np.zeros_like(mappings)
        mappings_inverse[:] = -1  # initialization
        for m, minv in zip(mappings, mappings_inverse):
            for ii, iv in enumerate(m):
                minv[iv] = ii
        return mappings_inverse
        

def get_rotations_cart(atoms):
    cell = atoms.get_cell()
    dataset = spglib.get_symmetry_dataset(atoms)
    rotations = dataset["rotations"]
    translations = dataset["translations"]

    rotations_cart = [
        np.dot(np.dot(cell.T, r), np.linalg.inv(cell.T)) for r in rotations
    ]
    rotations_cart = np.array(rotations_cart)

    return rotations_cart
