from abc import ABC, abstractmethod

from custom.interval import Interval


class Instance(ABC):
    def __init__(self):
        self.n_var = 0
        self.n_obj = 0
        self.n_constraints = 1

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
        self.umbral_indiferencia = []
        self.umbral_veto = []
        self.areas = []
        self.regiones = []

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
            self.umbral_indiferencia.append(float(v))
        index += 1
        for v in content[index].split():
            self.umbral_veto.append(float(v))
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
            self.regiones.append(a)
        index += 1
        self.n_var = int(content[index])
        for i in range(0, self.n_var):
            index += 1
            a = [float(v) for v in content[index].split()]
            self.projects.append(a)

    def __str__(self) -> str:
        return '{} {} {}'.format(super(PspInstance, self).__str__(), self.budget, self.weights)


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
            self.umbral_indiferencia.append(float(v))
        index += 1
        for v in content[index].split():
            self.umbral_veto.append(float(v))
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
            self.regiones.append(a)
        index += 1
        self.n_var = int(content[index])
        for i in range(0, self.n_var):
            index += 1
            a = [float(v) for v in content[index].split()]
            self.projects.append(a)
