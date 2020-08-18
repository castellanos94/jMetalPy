import random
from abc import ABC
from math import pi, cos
from typing import TypeVar

from custom.instance import DTLZInstance, Instance, PspInstance
from custom.interval import Interval
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution, BinarySolution

S = TypeVar('S')


class GDProblem(Problem[S], ABC):
    def __init__(self, instance_: Instance):
        super(GDProblem, self).__init__()
        self.instance_ = instance_
        self.number_of_variables = instance_.n_var
        self.number_of_objectives = instance_.n_obj
        self.number_of_constraints = instance_.n_constraints
        self.models = self.instance_.attributes['models']

    def get_preference_model(self, dm: int):
        return self.models[dm]


class BinaryProblemGD(GDProblem[BinarySolution], ABC):
    """ Class representing binary problems. """

    def __init__(self, instance_):
        super(BinaryProblemGD, self).__init__(instance_)
        self.number_of_bits = instance_.n_var
        self.number_of_variables = 1

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(
                self.number_of_bits)]

        return new_solution


class FloatProblemGD(GDProblem[FloatSolution], ABC):
    """ Class representing float problems. """

    def __init__(self, instance_):
        super(FloatProblemGD, self).__init__(instance_)
        self.lower_bound = []
        self.upper_bound = []

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.variables = \
            [random.uniform(self.lower_bound[index_var] * 1.0, self.upper_bound[index_var] * 1.0) for index_var in
             range(self.number_of_variables)]

        return new_solution


class DTLZ1Preferences(FloatProblemGD):
    """ Problem DTLZ1Preferences. Continuous problem having a flat Pareto front

        """

    def __init__(self, instance_: DTLZInstance):
        """ :param instance_: define number_of_variables and objectives also initial solution
        """
        super(DTLZ1Preferences, self).__init__(instance_)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(index_var) for index_var in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) - cos(20.0 * pi * (x - 0.5))
                 for x in solution.variables[self.number_of_variables - k:]])

        g = 100 * (k + g)

        solution.objectives = [(1.0 + g) * 0.5] * self.number_of_objectives

        for index_var in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (index_var + 1)):
                solution.objectives[index_var] *= solution.variables[j]

            if index_var != 0:
                solution.objectives[index_var] *= 1 - solution.variables[self.number_of_objectives - (index_var + 1)]

        return solution

    def get_name(self):
        return 'DTLZ1Preferences'


class PortfolioSocialProblem(BinaryProblemGD):
    def __init__(self, instance_: PspInstance):
        super(PortfolioSocialProblem, self).__init__(instance_)
        self.budget = instance_.budget

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        budget = 0
        objectives = self.number_of_objectives * [0.0]

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                budget += self.instance_.projects[index][0]
                for obj in range(0, self.number_of_objectives):
                    objectives[obj] += self.instance_.projects[index][obj + 3]
        solution.objectives = [-obj for obj in objectives]

        solution.constraints = [self.budget - budget]
        return solution

    def get_name(self) -> str:
        return 'PortfolioSocialProblem'


class PortfolioSocialProblemGD(BinaryProblemGD):
    def __init__(self, instance_: PspInstance):
        super(PortfolioSocialProblemGD, self).__init__(instance_)
        self.budget = instance_.budget
        self.positions = [idx for idx in range(self.number_of_bits)]

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = []
        budget = Interval(0)
        random.shuffle(self.positions)
        for v in self.positions:
            tmp = budget + self.instance_.projects[v][0]
            poss = self.budget.poss_greater_than_or_eq(tmp)
            if poss >= self.get_preference_model(0).chi:
                new_solution.variables[0].append(True)
                budget = tmp
            else:
                new_solution.variables[0].append(False)
        return new_solution

    def create_from_string(self, variables: str) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = \
            [True if variables[_] == '1' else False for _ in range(
                self.number_of_bits)]
        return new_solution

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        budget = Interval(0)
        objectives = self.number_of_objectives * [Interval(0)]
        for index, bits in enumerate(solution.variables[0]):
            if bits:
                budget += self.instance_.projects[index][0]
                for obj in range(0, self.number_of_objectives):
                    objectives[obj] += self.instance_.projects[index][obj+1]
        poss = self.budget.poss_greater_than_or_eq(budget)
        if poss < self.get_preference_model(0).chi:
            solution.constraints = [self.budget - budget]
        else:
            solution.constraints = [0]
        solution.objectives = [-obj for obj in objectives]
        return solution

    def get_name(self) -> str:
        return 'PortfolioSocialProblemGD'
