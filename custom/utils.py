import logging
import os

import numpy as np

from custom.instance import Instance
from custom.interval import Interval
from jmetal.algorithm.multiobjective.nsgaiii import ReferenceDirectionFactory
from jmetal.core.problem import Problem
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
    def __init__(self, problem: Problem, instance: Instance, normalize: bool = False):
        super(ReferenceDirectionFromSolution, self).__init__(n_dim=problem.number_of_objectives)
        self.problem = problem
        self.instance = instance
        self.normalize = normalize

    def _compute(self):
        ref_dir = []
        for s in self.instance.initial_solutions:
            self.problem.evaluate(s)
            ref_dir.append(np.array(s.objectives))
        if self.normalize:
            raise Exception('Not implemented yet.')
        # TODO: aqui hago algo
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
