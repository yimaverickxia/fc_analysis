#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO(ikeda): The structure of the variable "force_constants_pair" should be
#     modified. We want to use numpy functions.
from __future__ import absolute_import, division, print_function
import itertools
import numpy as np
from phonopy.structure.cells import Supercell
from fc_analysis.fc_distribution_analyzer import FCDistributionAnalyzer
from fc_analysis.fc_symmetrizer_spg import FCSymmetrizerSPG


class FCAnalyzer(object):
    def __init__(self,
                 force_constants=None,
                 atoms=None,
                 atoms_ideal=None,
                 supercell_matrix=None,
                 is_symmetrized=True):
        """

        Args:
            atoms: The "Atoms" object corresponding to the force constants.
        """
        if supercell_matrix is None:
            supercell_matrix = np.eye(3)

        print("supercell_matrix:")
        print(supercell_matrix)

        self.create_fc_symmetrizer_spg(
            force_constants, atoms, atoms_ideal, supercell_matrix)

        self.set_force_constants(force_constants)

        if atoms is not None:
            self.set_atoms(Supercell(atoms, supercell_matrix))
        if atoms_ideal is not None:
            self.set_atoms_ideal(Supercell(atoms_ideal, supercell_matrix))

        if is_symmetrized:
            self.symmetrize_force_constants()

        self._fc_distribution_analyzer = None

        self.check_consistency()

    def create_fc_symmetrizer_spg(
            self, force_constants, atoms, atoms_ideal, supercell_matrix):
        fc_symmetrizer_spg = FCSymmetrizerSPG(
            force_constants=force_constants,
            atoms=atoms,
            atoms_ideal=atoms_ideal,
            supercell_matrix=supercell_matrix,
        )
        self._fc_symmetrizer_spg = fc_symmetrizer_spg

    def check_consistency(self):
        number_of_atoms = self.get_atoms().get_number_of_atoms()
        number_of_atoms_fc = self.get_force_constants().shape[0]
        if number_of_atoms != number_of_atoms_fc:
            print(number_of_atoms, number_of_atoms_fc)
            raise ValueError("Atoms, Dim, and FC are not consistent.")

    def set_force_constants(self, force_constants):
        self._force_constants = force_constants
        return self

    def set_atoms(self, atoms):
        self._atoms = atoms

    def set_atoms_ideal(self, atoms_ideal):
        self._atoms_ideal = atoms_ideal

    def force_translational_invariance(self):
        # TODO(ikeda): Now only row blocks are considered.
        force_constants = self._force_constants
        for i in range(force_constants.shape[0]):
            tmp = np.sum(force_constants[i, :], axis=0)
            tmp -= force_constants[i, i]
            force_constants[i, i] = -tmp
        self.set_force_constants(force_constants)
        return self

    def check_translational_invariance(self):
        force_constants = self._force_constants
        for i in range(force_constants.shape[0]):
            print(i)
            print(np.sum(force_constants[i, :, :, :], axis=0))
        for i in range(force_constants.shape[1]):
            print(i)
            print(np.sum(force_constants[:, i, :, :], axis=0))

    def symmetrize_force_constants(self, iteration=3):
        symmetrize_force_constants(self._force_constants, iteration)
        return self

    def generate_dynamical_matrix(self):
        atoms = self._atoms
        natoms = atoms.get_number_of_atoms()
        masses = atoms.get_masses()
        dynamical_matrix = np.zeros(self._force_constants.shape) * np.nan
        for i1 in range(natoms):
            for i2 in range(natoms):
                m1 = masses[i1]
                m2 = masses[i2]
                dynamical_matrix[i1, i2] = (
                    self._force_constants[i1, i2] / np.sqrt(m1 * m2))
        self._dynamical_matrix = dynamical_matrix
        print("=" * 80)
        print(np.sum(self._force_constants, axis=0))
        print("=" * 80)
        print("=" * 80)
        print(np.sum(self._force_constants, axis=1))
        print("=" * 80)
        print("=" * 80)
        print(np.sum(self._dynamical_matrix, axis=0))
        print("=" * 80)
        print("=" * 80)
        print(np.sum(self._dynamical_matrix, axis=1))
        print("=" * 80)

    def normalize_force_constants(self):
        """Under testing...

        FORCE_CONSTANTS are overwritten.
        """
        atoms = self._atoms
        masses = atoms.get_masses()
        symbols = atoms.get_chemical_symbols()
        self._force_constants *= np.average(masses)
        self._force_constants_symmetrized *= np.average(masses)
        print(np.average(masses))
        self._force_constants_sd *= np.nan
        for s1 in symbols:
            for s2 in symbols:
                self._force_constants_pair[s1, s2] *= np.nan
                self._force_constants_pair_sd[s1, s2] *= np.nan

    def generate_effective_force_constants(self, atoms_symmetry=None, symprec=1e-5):
        self.generate_dynamical_matrix()
        self._force_constants = self._dynamical_matrix
        self.generate_symmetrized_force_constants(atoms_symmetry, symprec)
        self.normalize_force_constants()

    def set_atoms_ideal(self, atoms_ideal):
        self._atoms_ideal = atoms_ideal

    def analyze_fc_distribution(self,
                                a1,
                                a2,
                                filename="fc_values.dat",
                                symprec=1e-5):
        if self._fc_distribution_analyzer is None:
            self._fc_distribution_analyzer = FCDistributionAnalyzer(
                force_constants=self._force_constants,
                atoms=self._atoms,
                atoms_ideal=self._atoms_ideal,
                symprec=symprec,
            )
        self._fc_distribution_analyzer.analyze_fc_distribution(a1, a2, filename)

    def average_force_constants_spg(self, symprec=1e-5):
        self._fc_symmetrizer_spg.average_force_constants_spg(symprec=symprec)

    def generate_symmetrized_force_constants(self, symprec=1e-5):
        # TODO(ikeda): Too long method name
        """Generate symmetrized force constants.

        If the structure for extracting symmetry operations are different from
        the structure for extracting chemical symbols, we must specify symbols
        explicitly.
        """
        self._fc_symmetrizer_spg.average_force_constants_spg_full(symprec=symprec)

    def write_fc_for_specified_positions(self,
                                         pair_positions,
                                         filename_write="fc_values.dat",
                                         precision=15):

        iatom0 = self.get_index_from_position(pair_positions[0])
        iatom1 = self.get_index_from_position(pair_positions[1])

        self.write_fc_for_specified_indices((iatom0, iatom1),
                                            filename_write=filename_write,
                                            precision=precision)

    def write_fc_for_specified_indices(self,
                                       indices,
                                       filename_write="fc_values.dat",
                                       precision=15):

        width = precision + 7

        iatom0 = indices[0]
        iatom1 = indices[1]

        label_xyz = ["x", "y", "z"]

        with open(filename_write, "w") as f:

            def write_fc_tmp(fc0, fc1, i0, i1, s0, s1):
                f.write("{:4d}".format(iatom0))
                f.write("{:4d}".format(iatom1))
                f.write("  ")
                f.write("{:1s}".format(label_xyz[i0]))
                f.write("{:1s}".format(label_xyz[i1]))
                f.write("{:>8s}".format(s0))
                f.write("{:>8s}".format(s1))
                value = fc0[iatom0, iatom1, i0, i1]
                f.write("{:{width}.{precision}f}".format(
                    value,
                    width=width,
                    precision=precision))
                if fc1 is None:
                    value = np.nan
                else:
                    value = fc1[iatom0, iatom1, i0, i1]
                f.write("{:{width}.{precision}f}".format(
                    value,
                    width=width,
                    precision=precision))
                f.write("\n")

            for i0 in range(3):
                for i1 in range(3):
                    fc0 = self._force_constants
                    fc1 = None
                    write_fc_tmp(fc0, fc1, i0, i1, "", "")
                    for key in self._force_constants_pair.keys():
                        fc0 = self._force_constants_pair[key]
                        fc1 = self._force_constants_pair_sd[key]
                        write_fc_tmp(fc0, fc1, i0, i1, key[0], key[1])
                    fc0 = self._force_constants_symmetrized
                    fc1 = self._force_constants_sd
                    write_fc_tmp(fc0, fc1, i0, i1, "average", "average")
                    f.write("\n")

    def get_index_from_position(self, position, symprec=1e-6):

        for i, p in enumerate(self._atoms.get_scaled_positions()):
            diff = position - p
            diff -= np.rint(diff)
            if all([abs(x) < symprec for x in diff]):
                return i

