import random
from abc import ABC, abstractmethod
from typing import TypeVar, List

from custom.instance import Instance, PspInstance
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
        self.objectives_type = self.number_of_objectives * [False]  # Minimization

    def get_preference_model(self, dm: int):
        return self.models[dm]

    @abstractmethod
    def generate_existing_solution(self, variables) -> [S]:
        pass


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

    def generate_existing_solution(self, variables: str) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = \
            [True if variables[_] == '1' else False for _ in range(
                self.number_of_bits)]
        self.evaluate(new_solution)
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

    def generate_existing_solution(self, variables: List[float], is_objectives: bool = False) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        if not is_objectives:
            new_solution.variables = [variables[index_var] for index_var in range(self.number_of_variables)]
            self.evaluate(new_solution)
        else:
            new_solution.objectives = [variables[index_var] for index_var in range(self.number_of_objectives)]
        return new_solution


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
        self.objectives_type = self.number_of_objectives * [True]

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = []
        budget = Interval(0)
        random.shuffle(self.positions)
        new_solution.variables[0] = self.number_of_bits * [False]
        for v in self.positions:
            tmp = budget + self.instance_.projects[v][0]
            poss = self.budget.poss_greater_than_or_eq(tmp)
            if poss >= self.get_preference_model(0).chi:
                new_solution.variables[0][v] = True
                budget = tmp
        return new_solution

    def create_from_string(self, variables: str) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = \
            [True if variables[_] == '1' else False for _ in range(
                self.number_of_bits)]
        return new_solution

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        current_budget = Interval(0)
        objectives = self.number_of_objectives * [Interval(0)]
        for index, bits in enumerate(solution.variables[0]):
            if bits:
                current_budget += self.instance_.projects[index][0]
                for obj in range(0, self.number_of_objectives):
                    objectives[obj] += self.instance_.projects[index][obj + 1]
        poss = self.budget.poss_greater_than_or_eq(current_budget)
        if poss < self.get_preference_model(0).chi:
            solution.constraints = [self.budget - current_budget]
        else:
            solution.constraints = [0]
        solution.budget = current_budget
        solution.objectives = objectives
        return solution

    def get_name(self) -> str:
        return 'PortfolioSocialProblemGD'
