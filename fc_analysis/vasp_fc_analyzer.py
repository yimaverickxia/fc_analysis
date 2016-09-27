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


def run(args):
    fc_analyzer = VaspFCAnalyzer(
        force_constants_filename=args.force_constants,
        poscar=args.poscar,
        poscar_ideal=args.poscar_ideal,
        phonopy_conf=args.phonopy_conf,
    )

    # force_constants_analyzer.check_translational_invariance()

    if args.is_full:
        fc_analyzer.generate_symmetrized_force_constants()
    else:
        fc_analyzer.average_force_constants_spg()

    fc_analyzer.write_force_constants_symmetrized()
    fc_analyzer.write_force_constants_sd()
    fc_analyzer.write_force_constants_pair()
    fc_analyzer.write_force_constants_pair_sd()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force_constants",
                        default="FORCE_CONSTANTS",
                        type=str,
                        help="FORCE_CONSTANTS filename.")
    parser.add_argument("-p", "--poscar",
                        default="POSCAR",
                        type=str,
                        help="POSCAR filename.")
    parser.add_argument("--poscar_ideal",
                        # default="POSCAR_ideal",
                        type=str,
                        help="The filename of POSCAR_ideal.")
    parser.add_argument('--full', dest='is_full',
                        action='store_true',
                        help='Full FC symmetrization.')
    parser.add_argument("--phonopy_conf",
                        # default="POSCAR_ideal",
                        type=str,
                        required=True,
                        help="The filename of phonopy conf.")
    args = parser.parse_args()

    print(' '.join(sys.argv))

    run(args)

if __name__ == "__main__":
    main()
