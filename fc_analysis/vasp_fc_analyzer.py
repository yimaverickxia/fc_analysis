#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import sys
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from phonopy.interface.vasp import read_vasp
from fc_analysis.fc_analyzer import FCAnalyzer

class VaspFCAnalyzer(FCAnalyzer):
    def __init__(self,
                 force_constants_filename=None,
                 poscar=None,
                 poscar_ideal=None,
                 phonopy_conf=None,
                 is_symmetrized=True):

        if poscar_ideal is None:
            poscar_ideal = poscar

        force_constants = parse_FORCE_CONSTANTS(force_constants_filename)

        atoms = read_vasp(poscar)
        atoms_ideal = read_vasp(poscar_ideal)

        supercell_matrix = read_supercell_matrix(phonopy_conf)

        # force_constants_analyzer.generate_symmetrized_force_constants(
        #     atoms_symmetry)
        # force_constants_analyzer.write_force_constants_pair()

        super(VaspFCAnalyzer, self).__init__(
            force_constants=force_constants,
            atoms=atoms,
            atoms_ideal=atoms_ideal,
            supercell_matrix=supercell_matrix,
            is_symmetrized=True)


def read_supercell_matrix(phonopy_conf):
    from phonopy.cui.settings import PhonopyConfParser

    settings = PhonopyConfParser(phonopy_conf, option_list=[]).get_settings()
    return settings.get_supercell_matrix()
