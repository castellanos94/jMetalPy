from abc import ABC, abstractmethod

from custom.interval import Interval
from jmetal.core.solution import FloatSolution


class Instance(ABC):
    def __init__(self):
        self.n_var = 0
        self.n_obj = 0
        self.n_constraints = 1
        self.initial_solutions = []

    @abstractmethod
    def read_(self, absolute_path: str):
        pass

    def __str__(self) -> str:
        return '{} {} {}'.format(self.n_var, self.n_obj, self.n_constraints)


class PspInstance(Instance):
    def __init__(self):
        super().__init__()
        self.budget = 0
        self.weights = []
        self.projects = []
        self.indifference_threshold = []
        self.veto_threshold = []
        self.areas = []
        self.regions = []

    def read_(self, absolute_path: str):
        with open(absolute_path) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        index = 0
        self.budget = float(content[index])
        index += 1
        self.n_obj = int(content[index])
        index += 1
        for v in content[index].split():
            self.weights.append(float(v))
        index += 1
        for v in content[index].split():
            self.indifference_threshold.append(float(v))
        index += 1
        for v in content[index].split():
            self.veto_threshold.append(float(v))
        index += 1
        n_a = int(content[index])
        for i in range(0, n_a):
            index += 1
            a = [float(v) for v in content[index].split()]
            self.areas.append(a)
        index += 1
        n_r = int(content[index])
        for i in range(0, n_r):
            index += 1
            a = [float(v) for v in content[index].split()]
            self.regions.append(a)
        index += 1
        self.n_var = int(content[index])
        for i in range(0, self.n_var):
            index += 1
            a = [float(v) for v in content[index].split()]
            self.projects.append(a)

    def __str__(self) -> str:
        return '{} {} {}'.format(super(PspInstance, self).__str__(), self.budget, self.weights)


class DTLZInstance(Instance):

    def __init__(self):
        super().__init__()
        self.lower_bound = None
        self.upper_bound = None
        self.dms = 1
        self.weight = []
        self.veto = []
        self.lambdas = []
        self.initial_solutions = None
        self.best_compromise = None

    def read_(self, absolute_path: str):
        with open(absolute_path) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        index = 0
        self.n_obj = int(content[index].split()[0])
        index += 1
        self.n_var = int(content[index].split()[0])
        index += 1
        self.n_constraints = 0
        self.dms = int(content[index].split()[0])
        self.upper_bound = self.n_var * [0.0]
        self.lower_bound = self.n_var * [1.0]
        for dm in range(self.dms):
            index += 1
            line_split = content[index].split()
            idx = 0
            w = []
            while idx < self.n_obj * 2:
                lower = float(line_split[idx])
                idx += 1
                upper = float(line_split[idx].replace(',', ''))
                idx += 1
                w.append(Interval(lower, upper))
            self.weight.append(w)

        for dm in range(self.dms):
            index += 1
            line_split = content[index].split()
            idx = 0
            v = []
            while idx < self.n_obj * 2:
                lower = float(line_split[idx])
                idx += 1
                upper = float(line_split[idx].replace(',', ''))
                idx += 1
                v.append(Interval(lower, upper))
            self.veto.append(v)
        for i in range(self.dms):
            index += 1
            line_split = content[index].split()
            self.lambdas.append(Interval(float(line_split[0]), float(line_split[1].replace(',', ''))))
        index += 1
        if content[index].split()[0] == 'TRUE':
            index += 1
            n = int(content[index].split()[0])
            self.best_compromise = []

            for i in range(n):
                index += 1
                line_split = content[index].split()
                solution = FloatSolution(lower_bound=self.lower_bound, upper_bound=self.upper_bound,
                                         number_of_objectives=self.n_obj)
                solution.variables = [float(line_split[i].replace(',', '')) for i in range(0, self.n_var)]
                self.best_compromise.append(solution)
        index += 1
        if content[index].split()[0] == 'TRUE':
            index += 1
            n = int(content[index].split()[0])
            self.initial_solutions = []

            for i in range(n):
                index += 1
                line_split = content[index].split()
                solution = FloatSolution(lower_bound=self.lower_bound, upper_bound=self.upper_bound,
                                         number_of_objectives=self.n_obj)
                solution.variables = [float(line_split[i].replace(',', '')) for i in range(0, self.n_var)]
                self.initial_solutions.append(solution)


class PspIntervalInstance(PspInstance):
    def read_(self, absolute_path: str):
        with open(absolute_path) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        index = 0
        self.budget = Interval(content[index])
        index += 1
        self.n_obj = int(content[index])
        index += 1
        for v in content[index].split():
            self.weights.append(float(v))
        index += 1
        for v in content[index].split():
            self.indifference_threshold.append(float(v))
        index += 1
        for v in content[index].split():
            self.veto_threshold.append(float(v))
        index += 1
        n_a = int(content[index])
        for i in range(0, n_a):
            index += 1
            a = [float(v) for v in content[index].split()]
            self.areas.append(a)
        index += 1
        n_r = int(content[index])
        for i in range(0, n_r):
            index += 1
            a = [float(v) for v in content[index].split()]
            self.regions.append(a)
        index += 1
        self.n_var = int(content[index])
        for i in range(0, self.n_var):
            index += 1
            a = [float(v) for v in content[index].split()]
            self.projects.append(a)
