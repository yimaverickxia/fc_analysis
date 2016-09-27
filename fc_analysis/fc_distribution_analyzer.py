#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO(ikeda): The function "get_rotations_cart" is not appropriate for this
#     module.
from __future__ import absolute_import, division, print_function
import numpy as np
from fc_analysis.structure_analyzer import StructureAnalyzer
from fc_analysis.general_tools import get_rotations_cart
from fc_analysis.mappings_modifier import MappingsModifier


class FCDistributionAnalyzer(object):
    def __init__(self, force_constants, atoms, atoms_ideal, symprec=1e-5):
        self._force_constants = force_constants
        self._atoms = atoms
        self._atoms_ideal = atoms_ideal

        self.set_symprec(symprec)

        self._create_distance_matrix()
        self._create_symbol_numbers()
        self._create_rotations_cart()
        self._create_mappings()

    def set_symprec(self, symprec):
        self._symprec = symprec

    def _create_distance_matrix(self):
        sa = StructureAnalyzer(self._atoms)
        sa.generate_distance_matrix()
        self._distance_matrix = sa.get_distance_matrix()

    def _create_symbol_numbers(self):
        symbols = self._atoms.get_chemical_symbols()
        self._symbol_types, self._symbol_numbers = (
            SymbolNumbersGenerator().generate_symbol_numbers(symbols)
        )

    def _create_rotations_cart(self):
        self._rotations_cart = get_rotations_cart(self._atoms_ideal)

    def _create_mappings(self):
        # mappings: each index is for the "after" symmetry operations, and
        #     each element is for the "original" positions. 
        #     mappings[k][i] = j means the atom j moves to the positions of
        #     the atom i for the k-th symmetry operations.
        sa = StructureAnalyzer(self._atoms_ideal)
        mappings = sa.get_mappings_for_symops(prec=self._symprec)
        print("mappings: Finished.")
        mappings_inverse = MappingsModifier(mappings).invert_mappings()

        self._mappings = mappings
        self._mappings_inverse = mappings_inverse

    def analyze_fc_distribution(self,
                                a1,
                                a2,
                                filename="fc_values.dat"):

        symbol_numbers = self._symbol_numbers
        rotations_cart = self._rotations_cart
        mappings_inverse = self._mappings_inverse

        (nsym, natoms) = mappings_inverse.shape
        print("nsym: {}".format(nsym))
        print("natoms: {}".format(natoms))

        fc_values = np.zeros((nsym, 3, 3))
        fc_symbols = np.zeros((nsym, 2), dtype=int)
        distances = np.zeros(nsym)

        for isym, (minv, r) in enumerate(zip(mappings_inverse, rotations_cart)):
            # a1, a2: indices of atomic positions where i1 and i2 come
            # i1, i2: indices of atomic positions before symmetry operations
            i1 = minv[a1]
            i2 = minv[a2]
            i_s1 = symbol_numbers[i1]
            i_s2 = symbol_numbers[i2]

            fc_rotated = np.dot(np.dot(r, self._force_constants[i1, i2]), r.T)
            fc_values[isym] = fc_rotated
            fc_symbols[isym] = [i_s1, i_s2]
            distances[isym] = self._distance_matrix[i1, i2]

        self._print_fc_distribution(fc_symbols, fc_values, distances, filename)

    def _print_fc_distribution(self, fc_symbols, fc_values, distances, filename):
        with open(filename, "w") as f:
            for i, st in enumerate(self._symbol_types):
                f.write("# {:4d}: {:4s}\n".format(i, st))
            for si, d, v in zip(fc_symbols, distances, fc_values):
                f.write("{:4d}".format(si[0]))
                f.write("{:4d}".format(si[1]))
                f.write("{:22.15f}".format(d))
                f.write(" " * 4)
                for i in range(3):
                    for j in range(3):
                        f.write("{:22.15f}".format(v[i, j]))
                f.write("\n")


class SymbolNumbersGenerator(object):
    def generate_symbol_numbers(self, symbols):
        symbol_types = sorted(set(symbols), key=symbols.index)
        symbol_numbers = [symbol_types.index(s) for s in symbols]
        return symbol_types, symbol_numbers
