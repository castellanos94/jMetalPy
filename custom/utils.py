import logging
import os
import random
from typing import List

import numpy as np

from custom.gd_problems import GDProblem
from custom.interval import Interval
from jmetal.algorithm.multiobjective.nsgaiii import ReferenceDirectionFactory
from jmetal.core.solution import BinarySolution, Solution
from jmetal.util.comparator import DominanceComparator, Comparator, OverallConstraintViolationComparator

DIRECTORY_RESULTS = '/home/thinkpad/PycharmProjects/jMetalPy/results/'
LOGGER = logging.getLogger('jmetal')


def print_solutions_to_file(solutions, filename: str):
    filename = filename + '.csv'
    LOGGER.info('Output file (function values): ' + filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]
    binary = False
    if isinstance(solutions[0], BinarySolution):
        binary = True
    with open(filename, 'w') as of:
        of.write('variables * objectives * constraints\n')
        for solution in solutions:
            vars_str = ''
            if binary:
                for v in solution.variables[0]:
                    vars_str += '1' if v else '0'
            else:
                vars_str = str(solution.variables).replace('[', '').replace(']', '')
            of.write(vars_str + ' * ')
            of.write(str(solution.objectives).replace('[', '').replace(']', '') + ' * ')
            of.write(str(solution.constraints).replace('[', '').replace(']', '') + '\n')


class ReferenceDirectionFromSolution(ReferenceDirectionFactory):
    def __init__(self, problem: GDProblem, normalize: bool = False):
        super(ReferenceDirectionFromSolution, self).__init__(n_dim=problem.number_of_objectives)
        self.problem = problem
        self.instance = problem.instance_
        self.normalize = normalize

    def _compute(self):
        ref_dir = []
        for s in self.instance.initial_solutions:
            self.problem.evaluate(s)
            ref_dir.append(np.array(s.objectives))
        print(ref_dir)
        if self.normalize:
            min_f, max_f = np.min(ref_dir), np.max(ref_dir)
            norm = max_f - min_f
            ref_dir = (ref_dir - min_f) / norm

        return np.array(ref_dir)


class ITHDMDominanceComparator(DominanceComparator):
    """
    Eta-Dominance, default alpha value: 1.0
    """

    def __init__(self, alpha: float = 1, constraint_comparator: Comparator = OverallConstraintViolationComparator()):
        super().__init__(constraint_comparator)
        self.alpha = alpha

    def __dominance_test(self, solution1: Solution, solution2: Solution) -> float:
        best_is_one = 0
        best_is_two = 0
        value1_strictly_greater = False
        value2_strictly_greater = False
        for i in range(solution1.number_of_objectives):
            value1 = Interval(solution1.objectives[i])
            value2 = Interval(solution2.objectives[i])
            poss = value2.possibility(value1)
            if poss >= self.alpha:
                if not value1_strictly_greater and poss > 0.5:
                    value1_strictly_greater = True
                best_is_one += 1
            poss = value1.possibility(value2)
            if poss >= self.alpha:
                if not value2_strictly_greater and poss > 0.5:
                    value2_strictly_greater = True
                best_is_two += 1

        if value1_strictly_greater and best_is_one == solution1.number_of_variables:
            return -1
        if value2_strictly_greater and best_is_two == solution1.number_of_variables:
            return 1
        return 0


class DMGenerator:
    def __init__(self, number_of_variables: int, number_of_objectives: int, max_objectives: List[Interval]):
        self.numberOfObjectives = number_of_objectives
        self.numberOfVariables = number_of_variables
        self.maxObjectives = max_objectives

    def make(self):
        weights = self._generate_weights()
        veto = self._generate_veto(weights)
        return weights, veto

    def _generate_veto(self, weights: List[Interval]):
        v = []
        idx = 0
        while idx < self.numberOfObjectives:
            midp = self.maxObjectives[idx].midpoint()
            width = self.maxObjectives[idx].width()
            r1 = random.uniform(0, 1)
            vl = midp - r1 * (width / 10.0)
            r2 = random.uniform(0, 1)
            vu = midp + r2 * (width / 10.0)
            v.append(Interval(vl, vu))
            valid = True
            for jdx in range(idx):
                if weights[jdx] >= weights[idx] and v[jdx] >= v[idx]:
                    valid = False
                    break

            if valid:
                idx += 1
        return v

    def _generate_weights(self):
        weight = []
        valid = False
        while not valid:
            valid = True
            weight = self._butler_weight()
            if sum(weight) == 1.0:
                for v in weight:
                    if v > 0.5 * (1 - v):
                        valid = False
                        break
            else:
                valid = False
        return [Interval(w) for w in weight]

    def _butler_weight(self) -> list:
        vector = [0] * (self.numberOfObjectives + 1)
        same = True
        while same:
            vector = [random.randint(0, 1000) / 1000.0 for _ in vector]
            for idx in range(self.numberOfObjectives):
                while vector[idx] <= 0 or vector[idx] >= 1.0:
                    vector[idx] = random.randint(0, 1000) / 1000.0
            vector[self.numberOfObjectives] = 1
            same = False
            for idx in range(self.numberOfObjectives + 1):
                for jdx in range(self.numberOfObjectives):
                    if vector[jdx] > vector[jdx + 1]:
                        aux = vector[jdx]
                        vector[jdx] = vector[jdx + 1]
                        vector[jdx + 1] = aux
                    if vector[jdx] == vector[jdx + 1]:
                        same = True
        return vector[0:self.numberOfObjectives]
