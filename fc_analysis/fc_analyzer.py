#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO(ikeda): The function "get_rotations_cart" is not appropriate for this
#     module.
# TODO(ikeda): The structure of the variable "force_constants_pair" should be
#     modified. We want to use numpy functions.
from __future__ import absolute_import, division, print_function
import itertools
import numpy as np
from phonopy.structure import spglib
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.harmonic.force_constants import symmetrize_force_constants
from phonopy.structure.cells import Supercell
from fc_analysis.structure_analyzer import StructureAnalyzer


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
        self.set_force_constants(force_constants)

        if supercell_matrix is None:
            supercell_matrix = np.eye(3)

        print("supercell_matrix:")
        print(supercell_matrix)

        if atoms is not None:
            self.set_atoms(Supercell(atoms, supercell_matrix))
        if atoms_ideal is not None:
            self.set_atoms_ideal(Supercell(atoms_ideal, supercell_matrix))

        if is_symmetrized:
            self.symmetrize_force_constants()

        self._fc_distribution_analyzer = None

        self.check_consistency()

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

    def generate_symmetrized_force_constants(self, atoms_symmetry=None, symprec=1e-5):
        # TODO(ikeda): Too long method name
        """Generate symmetrized force constants.

        If the structure for extracting symmetry operations are different from
        the structure for extracting chemical symbols, we must specify symbols
        explicitly.
        """

        atoms = self._atoms
        symbols = atoms.get_chemical_symbols()
        symboltypes = sorted(set(symbols), key=symbols.index)
        nsymbols = len(symboltypes)

        if atoms_symmetry is None:
            atoms_symmetry = self._atoms

        # mappings: each index is for the "after" symmetry operations, and
        #     each element is for the "original" positions. 
        #     mappings[k][i] = j means the atom j moves to the positions of
        #     the atom i for the k-th symmetry operations.
        rotations_cart = get_rotations_cart(atoms_symmetry)
        mappings = StructureAnalyzer(
            atoms_symmetry).get_mappings_for_symops(prec=symprec)

        print("mappings: Finished.")
        (nsym, natoms) = mappings.shape
        print("nsym: {}".format(nsym))
        print("natoms: {}".format(natoms))

        shape = self._force_constants.shape

        force_constants_symmetrized = np.zeros(shape)
        force_constants_sd = np.zeros(shape)

        force_constants_pair = {}
        force_constants_pair_sd = {}
        pair_counters = {}
        for s1 in symboltypes:
            for s2 in symboltypes:
                force_constants_pair[(s1, s2)] = np.zeros(shape)
                force_constants_pair_sd[(s1, s2)] = np.zeros(shape)
                pair_counters[(s1, s2)] = np.zeros((natoms, natoms), dtype=int)

        for (m, r) in zip(mappings, rotations_cart):
            # i1, i2: indices after symmetry operations
            # j1, j2: indices before symmetry operations
            for i1 in range(natoms):
                for i2 in range(natoms):
                    j1 = m[i1]
                    j2 = m[i2]
                    s_i1 = symbols[i1]
                    s_i2 = symbols[i2]
                    s_j1 = symbols[j1]
                    s_j2 = symbols[j2]

                    tmp = np.dot(np.dot(r, self._force_constants[i1, i2]), r.T)
                    tmp2 = tmp ** 2
                    force_constants_symmetrized[j1, j2] += tmp
                    force_constants_sd[j1, j2] += tmp2

                    force_constants_pair[(s_i1, s_i2)][j1, j2] += tmp
                    force_constants_pair_sd[(s_i1, s_i2)][j1, j2] += tmp2
                    pair_counters[(s_i1, s_i2)][j1, j2] += 1

        self._pair_counters = pair_counters
        counter_check = np.zeros((natoms, natoms), dtype=int)
        for (key, c) in pair_counters.items():
            counter_check += c
        self._counter_check = counter_check

        def get_matrix_std(matrix_average, matrix_average_of_square):
            matrix_std = matrix_average_of_square - matrix_average ** 2
            return np.sqrt(matrix_std)

        force_constants_symmetrized /= float(nsym)
        force_constants_sd /= float(nsym)
        force_constants_sd = get_matrix_std(
            force_constants_symmetrized,
            force_constants_sd)

        for (s_i1, s_i2) in itertools.product(symboltypes, repeat=2):
            for (i1, i2) in itertools.product(range(natoms), repeat=2):
                cval = pair_counters[(s_i1, s_i2)][i1, i2]
                if cval != 0:
                    force_constants_pair[(s_i1, s_i2)][i1, i2] /= cval
                    force_constants_pair_sd[(s_i1, s_i2)][i1, i2] /= cval
                else:
                    force_constants_pair[(s_i1, s_i2)][i1, i2] = np.nan
                    force_constants_pair_sd[(s_i1, s_i2)][i1, i2] = np.nan
            force_constants_pair_sd[(s_i1, s_i2)] = get_matrix_std(
                force_constants_pair[(s_i1, s_i2)],
                force_constants_pair_sd[(s_i1, s_i2)])

        self._force_constants_symmetrized = force_constants_symmetrized
        self._force_constants_sd = force_constants_sd
        self._force_constants_pair = force_constants_pair
        self._force_constants_pair_sd = force_constants_pair_sd

    def get_force_constants(self):

        return self._force_constants

    def get_force_constants_symmetrized(self):

        return self._force_constants_symmetrized

    def get_force_constants_pair(self):

        return self._force_constants_pair

    def get_force_constants_sd(self):

        return self._force_constants_sd

    def get_atoms(self):

        return self._atoms

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

    def write_force_constants(self, filename_write):

        write_FORCE_CONSTANTS(self._force_constants, filename_write)

    def write_force_constants_symmetrized(
            self,
            filename_write="FORCE_CONSTANTS_SPG"):

        write_FORCE_CONSTANTS(self._force_constants_symmetrized,
                              filename_write)

    def write_force_constants_sd(
            self,
            filename_write="FORCE_CONSTANTS_SD"):

        write_FORCE_CONSTANTS(self._force_constants_sd,
                              filename_write)

    def write_force_constants_pair(
            self,
            filename_write="FORCE_CONSTANTS_PAIR"):

        for (pairtypes, force_constants_pair) in self._force_constants_pair.items():
            filename_write_pair = "{}_{}_{}".format(filename_write, *pairtypes)
            write_FORCE_CONSTANTS(force_constants_pair,
                                  filename_write_pair)

    def write_force_constants_pair_sd(
            self,
            filename_write="FORCE_CONSTANTS_PAIR_SD"):

        for (pairtypes, force_constants_pair_sd) in self._force_constants_pair_sd.items():
            filename_write_pair = "{}_{}_{}".format(filename_write, *pairtypes)
            write_FORCE_CONSTANTS(force_constants_pair_sd,
                                  filename_write_pair)

    def write_pair_counters(self, filename_write="PAIR_COUNTER"):

        for (pairtypes, pair_counter) in self._pair_counters.items():
            natoms = pair_counter.shape[0]
            filename_write_pair = "{}_{}_{}".format(filename_write, *pairtypes)
            with open(filename_write_pair, "w") as f:
                f.write("{:4d}\n".format(natoms))
                for (i1, i2) in itertools.product(range(natoms), repeat=2):
                    c = pair_counter[i1, i2]
                    f.write("{:4d}{:4d}{:8d}\n".format(i1, i2, c))

    def write_counter_check(self, filename_write="COUNTER_CHECK"):

        natoms = self._counter_check.shape[0]

        with open(filename_write, "w") as f:
            f.write("{:4d}\n".format(natoms))
            for (i1, i2) in itertools.product(range(natoms), repeat=2):
                c = self._counter_check[i1, i2]
                f.write("{:4d}{:4d}{:8d}\n".format(i1, i2, c))


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
        mappings_inverse = MappingsInverter().invert_mappings(mappings)

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
