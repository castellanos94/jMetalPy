import random
from abc import ABC
from math import pi, cos, sin
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
        self.objectives_type = self.number_of_objectives * [False]  # Minimization

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


class DTLZ1P(FloatProblemGD):
    """ Problem DTLZ1P. Continuous problem having a flat Pareto front

        """

    def __init__(self, instance_: DTLZInstance):
        """ :param instance_: define number_of_variables and objectives also initial solution
        """
        super(DTLZ1P, self).__init__(instance_)
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

    def generate_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.variables = []
        for _ in range(self.number_of_objectives - 1):
            new_solution.variables.append(random.random())
        for _ in range(self.number_of_objectives - 1, self.number_of_variables):
            new_solution.variables.append(0.5)
        self.evaluate(new_solution)
        return new_solution

    def get_name(self):
        return 'DTLZ1P_' + str(self.number_of_objectives)


class DTLZ2P(DTLZ1P):
    """ Problem DTLZ2. Continuous problem having a convex Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 12 and 3.
    """

    def __init__(self, instance_: DTLZInstance):
        """ :param instance_: define number_of_variables and objectives also initial solution
        """
        super(DTLZ2P, self).__init__(instance_)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(index_var) for index_var in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) for x in solution.variables[self.number_of_variables - k:]])

        solution.objectives = [1.0 + g] * self.number_of_objectives

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                solution.objectives[i] *= cos(solution.variables[j] * 0.5 * pi)

            if i != 0:
                solution.objectives[i] *= sin(0.5 * pi * solution.variables[self.number_of_objectives - (i + 1)])

        return solution

    def get_name(self):
        return 'DTLZ2P_' + str(self.number_of_objectives)


class DTLZ3P(DTLZ1P):
    """ Problem DTLZ3. Continuous problem having a convex Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 12 and 3.
    """

    def __init__(self, instance_: DTLZInstance):
        """ :param instance_: define number_of_variables and objectives also initial solution
        """
        super(DTLZ3P, self).__init__(instance_)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(index_var) for index_var in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum(
            [(x - 0.5) ** 2 - cos(20.0 * pi * (x - 0.5)) for x in solution.variables[self.number_of_variables - k:]])
        g = 100.0 * (k + g)

        f = [1.0 + g for _ in range(self.number_of_objectives)]

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                f[i] *= cos(solution.variables[j] * 0.5 * pi)

            if i != 0:
                aux = self.number_of_objectives - (i + 1)
                f[i] *= sin(solution.variables[aux] * 0.5 * pi)

        solution.objectives = [f[x] for x in range(self.number_of_objectives)]

        return solution

    def get_name(self):
        return 'DTLZ3P_' + str(self.number_of_objectives)


class DTLZ4P(DTLZ1P):
    """ Problem DTLZ4. Continuous problem having a convex Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 12 and 3.
    """

    def __init__(self, instance_: DTLZInstance):
        """ :param instance_: define number_of_variables and objectives also initial solution
        """
        super(DTLZ4P, self).__init__(instance_)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(index_var) for index_var in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        alpha = 100.0
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) ** 2 for x in solution.variables[self.number_of_variables - k:]])
        f = [1.0 + g for _ in range(self.number_of_objectives)]

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                f[i] *= cos(pow(solution.variables[j], alpha) * pi / 2.0)

            if i != 0:
                aux = self.number_of_objectives - (i + 1)
                f[i] *= sin(pow(solution.variables[aux], alpha) * pi / 2.0)

        solution.objectives = [f[x] for x in range(self.number_of_objectives)]

        return solution

    def get_name(self):
        return 'DTLZ4P_' + str(self.number_of_objectives)


class DTLZ5P(DTLZ1P):
    """ Problem DTLZ5. Continuous problem having a convex Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 12 and 3.
    """

    def __init__(self, instance_: DTLZInstance):
        """ :param instance_: define number_of_variables and objectives also initial solution
        """
        super(DTLZ5P, self).__init__(instance_)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(index_var) for index_var in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) ** 2 for x in solution.variables[self.number_of_variables - k:]])
        t = pi / (4.0 * (1.0 + g))

        theta = [0.0] * (self.number_of_objectives - 1)
        theta[0] = solution.variables[0] * pi / 2.0
        theta[1:] = [t * (1.0 + 2.0 * g * solution.variables[i]) for i in range(1, self.number_of_objectives - 1)]

        f = [1.0 + g for _ in range(self.number_of_objectives)]

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                f[i] *= cos(theta[j])

            if i != 0:
                aux = self.number_of_objectives - (i + 1)
                f[i] *= sin(theta[aux])

        solution.objectives = [f[x] for x in range(self.number_of_objectives)]

        return solution

    def get_name(self):
        return 'DTLZ5P_' + str(self.number_of_objectives)


class DTLZ6P(DTLZ1P):
    """ Problem DTLZ6. Continuous problem having a convex Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 12 and 3.
    """

    def __init__(self, instance_: DTLZInstance):
        """ :param instance_: define number_of_variables and objectives also initial solution
        """
        super(DTLZ6P, self).__init__(instance_)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(index_var) for index_var in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([pow(x, 0.1) for x in solution.variables[self.number_of_variables - k:]])
        t = pi / (4.0 * (1.0 + g))

        theta = [0.0] * (self.number_of_objectives - 1)
        theta[0] = solution.variables[0] * pi / 2.0
        theta[1:] = [t * (1.0 + 2.0 * g * solution.variables[i]) for i in range(1, self.number_of_objectives - 1)]

        f = [1.0 + g for _ in range(self.number_of_objectives)]

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                f[i] *= cos(theta[j])

            if i != 0:
                aux = self.number_of_objectives - (i + 1)
                f[i] *= sin(theta[aux])

        solution.objectives = [f[x] for x in range(self.number_of_objectives)]

        return solution

    def get_name(self):
        return 'DTLZ6P_' + str(self.number_of_objectives)


class DTLZ7P(DTLZ1P):
    """ Problem DTLZ6. Continuous problem having a disconnected Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 22 and 3.
    """

    def __init__(self, instance_: DTLZInstance):
        """ :param instance_: define number_of_variables and objectives also initial solution
        """
        super(DTLZ7P, self).__init__(instance_)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(index_var) for index_var in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([x for x in solution.variables[self.number_of_variables - k:]])
        g = 1.0 + (9.0 * g) / k

        h = sum([(x / (1.0 + g)) * (1 + sin(3.0 * pi * x)) for x in solution.variables[:self.number_of_objectives - 1]])
        h = self.number_of_objectives - h

        solution.objectives[:self.number_of_objectives - 1] = solution.variables[:self.number_of_objectives - 1]
        solution.objectives[-1] = (1.0 + g) * h

        return solution

    def get_name(self):
        return 'DTLZ7P_' + str(self.number_of_objectives)


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
