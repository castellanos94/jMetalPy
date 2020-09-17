from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from custom.interval import Interval
from custom.utils import OutrankingModel
from jmetal.core.solution import FloatSolution


class Instance(ABC):
    def __init__(self):
        self.attributes = {'models': []}
        self.n_var = 0
        self.n_obj = 0
        self.n_constraints = 1
        self.initial_solutions = []
        self.dms = 1

    @abstractmethod
    def read_(self, absolute_path: str) -> Instance:
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

    def read_(self, absolute_path: str) -> PspInstance:
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
        return self

    def __str__(self) -> str:
        return '{} {} {}'.format(super(PspInstance, self).__str__(), self.budget, self.weights)


class DTLZInstance(Instance):

    def __init__(self):
        super().__init__()
        self.lower_bound = None
        self.upper_bound = None
        self.dms = 1
        self.initial_solutions = None

    def read_(self, absolute_path: str) -> DTLZInstance:
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
        weight = []
        for dm in range(self.dms):
            index += 1
            line_split = clean_line(content[index])
            idx = 0
            w = []
            while idx < self.n_obj * 2:
                lower = float(line_split[idx])
                idx += 1
                upper = float(line_split[idx].replace(',', ''))
                idx += 1
                w.append(Interval(lower, upper))
            weight.append(w)
        veto = []
        for dm in range(self.dms):
            index += 1
            line_split = clean_line(content[index])
            idx = 0
            v = []
            while idx < self.n_obj * 2:
                lower = float(line_split[idx])
                idx += 1
                upper = float(line_split[idx].replace(',', ''))
                idx += 1
                v.append(Interval(lower, upper))
            veto.append(v)
        beta = []
        for i in range(self.dms):
            index += 1
            line_split = content[index].split()
            beta.append(Interval(float(line_split[0]), float(line_split[1].replace(',', ''))))
        lambdas = []
        for i in range(self.dms):
            index += 1
            line_split = content[index].split()
            lambdas.append(Interval(float(line_split[0]), float(line_split[1].replace(',', ''))))
        index += 1
        if content[index].split()[0] == 'TRUE':
            index += 1
            n = int(content[index].split()[0])
            best_compromise = []

            for i in range(n):
                index += 1
                line_split = content[index].split()
                best_compromise.append([float(line_split[i].replace(',', '')) for i in range(0, self.n_var)])
            self.attributes['best_compromise'] = best_compromise
        index += 1
        line = clean_line(content[index])
        if line[0] == "TRUE":
            r2_set = []
            r1_set = []
            frontiers = []
            for dm in range(self.dms):
                index += 1
                line = clean_line(content[index])
                n_size = int(int(line[0]) / 2)
                r2 = []
                r1 = []
                for n_row in range(n_size):
                    r2_ = []
                    index += 1
                    line = clean_line(content[index])
                    idx = 0
                    while idx < self.n_obj * 2:
                        r2_.append(Interval(line[idx], line[idx + 1]))
                        idx += 2
                    r2.append(r2_)
                for n_row in range(n_size):
                    r1_ = []
                    index += 1
                    line = clean_line(content[index])
                    idx = 0
                    while idx < self.n_obj * 2:
                        r1_.append(Interval(line[idx], line[idx + 1]))
                        idx += 2
                    r1.append(r1_)
                r2_set.append(r2)
                r1_set.append(r1)
                frontiers.append(r2 + r1)
            self.attributes['frontiers'] = frontiers
            self.attributes['r2'] = r2_set
            self.attributes['r1'] = r1_set

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
        self.attributes['dms'] = self.dms
        models = []
        for dm in range(self.dms):
            model = OutrankingModel(weight[dm], veto[dm], 1.0, beta[dm], lambdas[dm], 1)
            models.append(model)
        self.attributes['models'] = models
        return self


def clean_line(line: str) -> List[str]:
    line_ = line.replace('\"', "").split("//")
    line_ = [v.replace(',', ' ') for v in line_[0].split()]
    rs = []
    for v in line_:
        rs += v.split()
    return rs


class PspIntervalInstance(PspInstance):

    def read_(self, absolute_path: str) -> PspIntervalInstance:
        with open(absolute_path) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        index = 0
        line = clean_line(content[index])
        self.budget = Interval(line[0], line[1])
        index += 1
        line = clean_line(content[index])
        self.n_obj = int(line[0])
        index += 1
        line = clean_line(content[index])
        self.attributes['dms'] = int(line[0])
        dm_type = []
        for idx in range(self.attributes['dms']):
            index += 1
            line = clean_line(content[index])
            dm_type.append(int(line[0]))
        index += 1
        line = clean_line(content[index])
        self.n_constraints = int(line.__getitem__(0))

        for dm in range(self.attributes['dms']):
            index += 1
            line = clean_line(content[index])
            idx = 0
            w = []
            while idx < self.n_obj * 2:
                w.append(Interval(line[idx], line[idx + 1]))
                idx += 2
            self.weights.append(w)
        for dm in range(self.attributes['dms']):
            index += 1
            line = clean_line(content[index])
            idx = 0
            v = []
            while idx < self.n_obj * 2:
                v.append(Interval(line[idx], line[idx + 1]))
                idx += 2
            self.veto_threshold.append(v)
        beta = []
        for dm in range(self.attributes['dms']):
            index += 1
            line = clean_line(content[index])
            beta.append(Interval(line[0], line[1]))
        chi = []
        for dm in range(self.attributes['dms']):
            index += 1
            line = clean_line(content[index])
            chi.append(float(line[0]))
        alpha = []
        for dm in range(self.attributes['dms']):
            index += 1
            line = clean_line(content[index])
            alpha.append(float(line[0]))
        lambdas = []
        for dm in range(self.attributes['dms']):
            index += 1
            line = clean_line(content[index])
            lambdas.append(Interval(line[0], line[1]))
        models = []
        for dm in range(self.attributes['dms']):
            models.append(OutrankingModel(
                self.weights[dm], self.veto_threshold[dm], alpha[dm], beta[dm], lambdas[dm], chi[dm], dm_type[dm] == 1
            ))
        self.attributes['models'] = models
        index += 1
        line = clean_line(content[index])
        self.n_var = int(line[0])
        self.projects = []
        for p in range(self.n_var):
            idx = 0
            index += 1
            line = clean_line(content[index])
            project = []
            while idx < self.n_obj * 2 + 1:
                project.append(Interval(line[idx], line[idx + 1]))
                idx += 2
            self.projects.append(project)
        index += 1
        line = clean_line(content[index])
        if line[0] == "TRUE":
            best_compromise = []
            for dm in range(self.attributes['dms']):
                v = ''
                index += 1
                line = clean_line(content[index])
                for idx in range(self.n_var):
                    v += line[idx]
                best_compromise.append(v)
            self.attributes['best_compromise'] = best_compromise
        index += 1
        line = clean_line(content[index])
        if line[0] == "TRUE":
            r2_set = []
            r1_set = []
            frontiers = []
            for dm in range(self.attributes['dms']):
                index += 1
                line = clean_line(content[index])
                n_size = int(int(line[0]) / 2)
                r2 = []
                r1 = []
                for n_row in range(n_size):
                    r2_ = []
                    index += 1
                    line = clean_line(content[index])
                    idx = 0
                    while idx < self.n_obj * 2:
                        r2_.append(Interval(line[idx], line[idx + 1]))
                        idx += 2
                    r2.append(r2_)
                for n_row in range(n_size):
                    r1_ = []
                    index += 1
                    line = clean_line(content[index])
                    idx = 0
                    while idx < self.n_obj * 2:
                        r1_.append(Interval(line[idx], line[idx + 1]))
                        idx += 2
                    r1.append(r1_)
                r2_set.append(r2)
                r1_set.append(r1)
                frontiers.append(r2 + r1)
            self.attributes['frontiers'] = frontiers
            self.attributes['r2'] = r2_set
            self.attributes['r1'] = r1_set
        return self
