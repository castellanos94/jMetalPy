import random
from math import pi, cos

import numpy as np

from custom.instance import DTLZInstance
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.lab.visualization import Plot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.solution import print_solution_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations


class DTLZ1Preferences(FloatProblem):
    """ Problem DTLZ1. Continuous problem having a flat Pareto front

        .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 7 and 3.
        """

    def __init__(self, instance: DTLZInstance):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super(DTLZ1Preferences, self).__init__()
        self.number_of_variables = instance.n_var
        self.number_of_objectives = instance.n_obj
        self.number_of_constraints = 0
        self.instance = instance
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) - cos(20.0 * pi * (x - 0.5))
                 for x in solution.variables[self.number_of_variables - k:]])

        g = 100 * (k + g)

        solution.objectives = [(1.0 + g) * 0.5] * self.number_of_objectives

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                solution.objectives[i] *= solution.variables[j]

            if i != 0:
                solution.objectives[i] *= 1 - solution.variables[self.number_of_objectives - (i + 1)]

        return solution

    def get_name(self):
        return 'DTLZ1Preferences'


if __name__ == '__main__':
    random.seed(8435)
    instance = DTLZInstance()
    path = '/home/thinkpad/Documents/jemoa/src/main/resources/instances/dtlz/DTLZInstance.txt'
    instance.read_(path)
    problem = DTLZ1Preferences(instance)
    ref_dir = []
    for s in instance.initial_solutions:
        problem.evaluate(s)
        ref_dir.append(np.array(s.objectives))
    ref_dir = np.array(ref_dir)
    print(len(instance.initial_solutions))
    # problem.reference_front = read_solutions(filename='resources/reference_front/DTLZ2.3D.pf')

    max_evaluations = 25000
    experiment = 50
    algorithm = NSGAIII(
        problem=problem,
        population_size=100,
        reference_directions=ref_dir,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=30),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )
    bag = []
    for i in range(experiment):
        algorithm = NSGAIII(
            problem=problem,
            population_size=92,
            reference_directions=ref_dir,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=30),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )
        progress_bar = ProgressBarObserver(max=max_evaluations)
        algorithm.observable.register(progress_bar)
        algorithm.run()

        bag = bag + algorithm.get_result()
    print(len(bag))

    ranking = FastNonDominatedRanking()
    print_solution_to_file(bag, 'Solutions.bag.' + algorithm.label)

    ranking.compute_ranking(bag)
    front = ranking.get_subfront(0)
    print(len(front))
    # Save results to file
    print_solution_to_file(front, 'Solutions.front0.' + algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
    plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y', 'z'])
    plot_front.plot(front, label='NSGAII-ZDT1-preferences_bag', filename='NSGAII-ZDT1-p_f0', format='png')
