import random
from math import cos, pi, sin

from custom.gd_problems import FloatProblemGD
from custom.instance import DTLZInstance
from custom.util_problem import InterClassNC
from jmetal.core.solution import FloatSolution


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
        self.classifier = InterClassNC(self)
        self.number_of_constraints = 4

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

        c = self.classifier.classify(solution)
        c[2] = -c[2]
        c[3] = -c[3]
        solution.constraints = c
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
