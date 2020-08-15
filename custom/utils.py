import logging
import os

import numpy as np

from custom.instance import Instance
from jmetal.algorithm.multiobjective.nsgaiii import ReferenceDirectionFactory
from jmetal.core.problem import Problem
from jmetal.core.solution import BinarySolution

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
