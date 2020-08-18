import logging
import os
import random
from typing import List

import numpy as np

from custom.interval import Interval
from jmetal.algorithm.multiobjective.nsgaiii import ReferenceDirectionFactory
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


class OutrankingModel:
    def __init__(self, weights: List[Interval], veto: List[Interval], alpha, beta, lambda_, chi,
                 supports_utility_function: bool = False):
        self.weights = weights
        self.veto = veto
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.chi = chi
        self.supports_utility_function = supports_utility_function
        self.dominance_comparator = ITHDMDominanceComparator()

class DMGenerator:
    def __init__(self, number_of_variables: int, number_of_objectives: int, max_objectives: List[Interval]):
        self.numberOfObjectives = number_of_objectives
        self.numberOfVariables = number_of_variables
        self.maxObjectives = max_objectives

    def make(self):
        weights = self._generate_weights()
        veto = self._generate_veto(weights)
        return weights, veto

    def _generate_veto(self, weights: List[Interval]):
        v = []
        idx = 0
        while idx < self.numberOfObjectives:
            midp = self.maxObjectives[idx].midpoint()
            width = self.maxObjectives[idx].width()
            r1 = random.uniform(0, 1)
            vl = midp - r1 * (width / 10.0)
            r2 = random.uniform(0, 1)
            vu = midp + r2 * (width / 10.0)
            v.append(Interval(vl, vu))
            valid = True
            for jdx in range(idx):
                if weights[jdx] >= weights[idx] and v[jdx] >= v[idx]:
                    valid = False
                    break

            if valid:
                idx += 1
        return v

    def _generate_weights(self):
        weight = []
        valid = False
        while not valid:
            valid = True
            weight = self._butler_weight()
            if sum(weight) == 1.0:
                for v in weight:
                    if v > 0.5 * (1 - v):
                        valid = False
                        break
            else:
                valid = False
        return [Interval(w) for w in weight]

    def _butler_weight(self) -> list:
        vector = [0] * (self.numberOfObjectives + 1)
        same = True
        while same:
            vector = [random.randint(1, 10000) / 10000.0 for _ in vector]
            for idx in range(self.numberOfObjectives):
                while vector[idx] <= 0 or vector[idx] >= 1.0:
                    vector[idx] = random.randint(1, 10000) / 10000.0
            vector[self.numberOfObjectives] = 1
            same = False
            for idx in range(self.numberOfObjectives + 1):
                for jdx in range(self.numberOfObjectives):
                    if vector[jdx] > vector[jdx + 1]:
                        aux = vector[jdx]
                        vector[jdx] = vector[jdx + 1]
                        vector[jdx + 1] = aux
                    if vector[jdx] == vector[jdx + 1]:
                        same = True
        return vector[0:self.numberOfObjectives]


class ITHDMPreferences:
    """
    This class determines what kind of outranking relationship exists between two
    solutions: x, y. Fernández,J.R.FigueiraandJ.Navarro,Interval-based extensions
    of two outranking methods for multi-criteria ordinal classification, Omega,
    https://doi.org/10.1016/j.omega.2019.05.001
    """

    def __init__(self, preference_model: OutrankingModel):
        self.preference_model = preference_model
        self.sigmaXY = None
        self.sigmaYX = None
        self.coalition = None

    """
        Definition 3. Relationships:xS(δ,λ)y in [-2], xP(δ,λ)y in [-1], xI(δ,λ)y in [0], xR(δ,λ)y in [1]
    """

    def compare(self, x: Solution, y: Solution) -> int:
        self.coalition = [None for _ in range(x.number_of_objectives)]
        self.sigmaXY = self._credibility_index(x.objectives, y.objectives)
        self.sigmaYX = self._credibility_index(y.objectives, x.objectives)
        if self.preference_model.dominance_comparator.compare(x, y) == -1:
            return -2
        if self.sigmaXY >= self.preference_model.beta > self.sigmaYX:
            return -1
        if self.sigmaXY >= self.preference_model.beta and self.sigmaYX >= self.preference_model.beta:
            return 0
        if self.sigmaXY < self.preference_model.beta and self.sigmaYX < 0:
            return 1
        return 2

    def _credibility_index(self, x: List, y: List) -> float:
        omegas = []
        dj = []
        eta_gamma = [0] * len(x)
        max_eta_gamma = float('-inf')
        for idx in range(len(x)):
            omegas.append(self._alpha_ij(x, y, idx))
            dj.append(self._discordance_ij(x, y, idx))
        for idx in range(len(x)):
            gamma = omegas[idx]
            ci = self._concordance_index(gamma, omegas)
            poss = ci.poss_greater_than_or_eq(self.preference_model.lambda_)
            max_discordance = float('-inf')
            for jdx in range(len(x)):
                if self.coalition[jdx] == 0 and dj[jdx] > max_discordance:
                    max_discordance = dj[jdx]
            non_discordance = 1 - max_discordance
            eta_gamma[idx] = gamma
            if eta_gamma[idx] > poss:
                eta_gamma[idx] = poss
            if eta_gamma[idx] > non_discordance:
                eta_gamma[idx] = non_discordance
            if max_eta_gamma < eta_gamma[idx]:
                max_eta_gamma = eta_gamma[idx]
        return max_eta_gamma

    def _concordance_index(self, gamma: float, omegas: List) -> Interval:
        cl = 0
        cu = 0
        dl = 0
        du = 0
        i_weights = self.preference_model.weights
        for idx in range(len(i_weights)):
            if omegas[idx] >= gamma:
                self.coalition[idx] = 1
                cl += i_weights[idx].lower
                cu += i_weights[idx].upper
            else:
                self.coalition[idx] = 0
                dl += i_weights[idx].lower
                du += i_weights[idx].upper
        if (cl + du) >= 1:
            lower = cl
        else:
            lower = 1 - du
        if (cu + dl) <= 1:
            upper = cu
        else:
            upper = 1 - dl
        return Interval(lower, upper)

    def _discordance_ij(self, x: List[Interval], y: List[Interval], criteria: int) -> float:
        veto = self.preference_model.veto[criteria]
        return y[criteria].poss_smaller_than_or_eq(x[criteria] + veto)

    @staticmethod
    def _alpha_ij(x: List[Interval], y: List[Interval], criteria: int) -> float:
        return x[criteria].poss_smaller_than_or_eq(y[criteria])


class ITHDMPreferenceUF:
    """
    Pendiente revisar, implmentacion base rara
    """

    def __init__(self, preference_model: OutrankingModel):
        self.preference_model = preference_model

    """
    -1 if xPy, 0 if x~y, 1 otherwise
    """

    def compare(self, x: Solution, y: Solution) -> int:
        if self.preference_model.dominance_comparator.compare(x, y) == -1:
            return -1
        ux = Interval(0)
        uy = Interval(0)
        for idx in range(x.number_of_objectives):
            ux += self.preference_model.weights[idx] * x.objectives[idx]
            uy += self.preference_model.weights[idx] * y.objectives[idx]
        if ux >= uy:
            return -1
        return 0



class BestCompromiseDTLZ:
    def __init__(self):
        pass
