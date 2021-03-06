import logging
import os
import random
from typing import List, Tuple

from custom.interval import Interval
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

    def __init__(self, objectives_type: List[bool], alpha: float = 1,
                 constraint_comparator: Comparator = OverallConstraintViolationComparator()):
        super().__init__(constraint_comparator)
        self.alpha = alpha
        self.objectives_type = objectives_type

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            raise Exception("The solution1 is None")
        elif solution2 is None:
            raise Exception("The solution2 is None")

        result = self.constraint_comparator.compare(solution1, solution2)
        if result == 0:
            result = self.dominance_test_(solution1.objectives, solution2.objectives)
            # result = self.dominance_test(solution1.objectives, solution2.objectives)

        return result

    def dominance_test_(self, solution1: List, solution2: List) -> float:
        best_is_one = 0
        best_is_two = 0
        value1_strictly_greater = False
        value2_strictly_greater = False

        for i in range(len(solution1)):
            value1 = Interval(solution1[i])
            value2 = Interval(solution2[i])
            if self.objectives_type[i]:
                poss = value1.possibility(value2)
            else:
                poss = value2.possibility(value1)
            if poss >= self.alpha:
                if not value1_strictly_greater and poss > 0.5:
                    value1_strictly_greater = True
                best_is_one += 1
            if self.objectives_type[i]:
                poss = value2.possibility(value1)
            else:
                poss = value1.possibility(value2)
            if poss >= self.alpha:
                if not value2_strictly_greater and poss > 0.5:
                    value2_strictly_greater = True
                best_is_two += 1
        s1_dominates_s2 = False
        s2_dominates_s1 = False
        if value1_strictly_greater and best_is_one == len(solution1):
            s1_dominates_s2 = True
        if value2_strictly_greater and best_is_two == len(solution1):
            s2_dominates_s1 = True
        if s2_dominates_s1 and s1_dominates_s2:
            return 0
        if s1_dominates_s2:
            return -1
        if s2_dominates_s1:
            return 1
        if not s2_dominates_s1 and not s1_dominates_s2 and best_is_one != len(solution1) and best_is_two != len(
                 solution1):
             if self.objectives_type[0]: # Max
                 if best_is_one < best_is_two:
                     return -1
                 if best_is_two < best_is_one:
                     return 1
             else:
                 if best_is_one > best_is_two:
                     return -1
                 if best_is_two > best_is_one:
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

    def __str__(self) -> str:
        return "weights : {}, veto : {}, alpha : {}, beta : {}, lambda : {}, chi : {}, uf : {}".format(self.weights,
                                                                                                       self.veto,
                                                                                                       self.alpha,
                                                                                                       self.beta,
                                                                                                       self.lambda_,
                                                                                                       self.chi,
                                                                                                       self.supports_utility_function)

    def __repr__(self):
        return self.__str__()


class DMGenerator:
    def __init__(self, number_of_variables: int, number_of_objectives: int, max_objectives: List[Interval]):
        self.numberOfObjectives = number_of_objectives
        self.numberOfVariables = number_of_variables
        self.maxObjectives = max_objectives

    def make(self) -> Tuple[List[Interval], List[Interval]]:
        print("Generating weights...")
        weights = self._generate_weights()
        print("Generating veto with: ", weights)
        veto = self._generate_veto(weights)
        return weights, veto

    def _generate_veto(self, weights: List[Interval]):
        v = []
        min_, max_ = min(weights), max(weights)
        weights_norm = [(v_ - min_) / (max_ - min_) for v_ in weights]
        idx = 0
        while idx < self.numberOfObjectives:
            midp = self.maxObjectives[idx].midpoint()
            width = self.maxObjectives[idx].width()
            r1 = random.uniform(0, 1)
            vl = midp - r1 * (width / 10.0)
            r2 = random.uniform(0, 1)
            vu = midp + r2 * (width / 10.0)
            v.append(Interval(vl, vu))
            valid = False
            for jdx in range(idx):
                if weights_norm[jdx] >= weights_norm[idx] and v[jdx] >= v[idx]:
                    valid = True

            if not valid:
                idx += 1
        # print("before re-normalize:", v)
        # return [(value + min_) * (max_ - min_) for value in v]
        return v

    def _generate_weights(self):
        weight = []
        valid = False
        while not valid:
            valid = True
            weight = self._butler_weight()
            if sum(weight) == 1.0:
                if self.numberOfObjectives == 3:
                    for _ in range(self.numberOfObjectives - 1):
                        if weight[_] > 0.5 * (1 - weight[_]):
                            valid = False
                            break
                    if weight[self.numberOfObjectives - 1] > 0.6 * (1 - weight[self.numberOfObjectives - 1]):
                        valid = False
                else:
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
            vector = [random.randint(1, 1000) / 1000.0 for _ in vector]
            for idx in range(self.numberOfObjectives):
                while vector[idx] <= 0 or vector[idx] >= 1.0:
                    vector[idx] = random.randint(1, 1000) / 1000.0
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

    def __init__(self, objectives_type: List[bool], preference_model: OutrankingModel):
        self.preference_model = preference_model
        self.sigmaXY = None
        self.sigmaYX = None
        self.coalition = None
        self.objectives_type = objectives_type
        self.dominance_comparator = ITHDMDominanceComparator(objectives_type, preference_model.alpha)

    def compare(self, x: Solution, y: Solution) -> int:
        """
            Definition 3. Relationships:xS(δ,λ)y in [-2], xP(δ,λ)y in [-1], xI(δ,λ)y in [0], xR(δ,λ)y in [1]

        """
        return self.compare_(x.objectives, y.objectives)

    def compare_(self, x: List[Interval], y: List[Interval]) -> int:
        self.coalition = [None for _ in range(len(x))]
        self.sigmaXY = self._credibility_index(x, y)
        self.sigmaYX = self._credibility_index(y, x)
        value = self.dominance_comparator.dominance_test_(x, y)
        if value == -1:
            return -2
        if value == 1:
            return 2

        if self.sigmaXY >= self.preference_model.beta > self.sigmaYX:
            return -1
        if self.sigmaXY >= self.preference_model.beta and self.sigmaYX >= self.preference_model.beta:
            return 0
        if self.sigmaXY < self.preference_model.beta and self.sigmaYX < 0:
            return 1
        return 1

    def _credibility_index(self, x: List[Interval], y: List[Interval]) -> float:
        omegas = [0] * len(x)
        dj = [0] * len(x)
        eta_gamma = [0] * len(x)
        max_eta_gamma = float('-inf')
        for idx in range(len(x)):
            omegas[idx] = self._alpha_ij(x, y, idx)
            dj[idx] = self._discordance_ij(x, y, idx)

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
        if self.objectives_type[criteria]:
            return Interval(y[criteria]).poss_greater_than_or_eq(x[criteria] + veto)
        return Interval(y[criteria]).poss_smaller_than_or_eq(x[criteria] - veto)

    def _alpha_ij(self, x: List[Interval], y: List[Interval], criteria: int) -> float:
        if self.objectives_type[criteria]:
            return Interval(x[criteria]).poss_greater_than_or_eq(y[criteria])
        return Interval(x[criteria]).poss_smaller_than_or_eq(y[criteria])


class ITHDMPreferenceUF:
    """
    Pendiente revisar, implmentacion base rara
    """

    def __init__(self, objectives_type: List[bool], preference_model: OutrankingModel):
        self.preference_model = preference_model
        self.objectives_type = objectives_type
        self.dominance_comparator = ITHDMDominanceComparator(objectives_type, preference_model.alpha)

    def compare(self, x: Solution, y: Solution) -> int:
        """
            -1 if xPy, 0 if x~y, 1 otherwise
        """
        if self.dominance_comparator.compare(x, y) == -1:
            return -1
        ux = Interval(0)
        uy = Interval(0)
        for idx in range(x.number_of_objectives):
            ux += self.preference_model.weights[idx] * x.objectives[idx]
            uy += self.preference_model.weights[idx] * y.objectives[idx]
        if ux >= uy:
            return -1
        if ux < uy:
            return 1
        return 0

    def compare_(self, x: List, y: List) -> int:
        """
                    -1 if xPy, 0 if x~y, 1 otherwise
                """
        if self.dominance_comparator.dominance_test_(x, y) == -1:
            return -1
        ux = Interval(0)
        uy = Interval(0)
        for idx in range(len(x)):
            ux += self.preference_model.weights[idx] * x[idx]
            uy += self.preference_model.weights[idx] * y[idx]
        if ux >= uy:
            return -1
        if ux < uy:
            return 1
        return 0


class ITHDMRanking:

    def __init__(self, preferences, a_pref: List[int], b_pref: List[int]):
        self.preferences = preferences
        self.number_of_comparisons = 0
        self.a_pref = a_pref
        self.b_pref = b_pref

    def compute_ranking(self, solutions: List[Solution], k: int = None):
        """ Compute ranking of solutions.

        :param solutions: Solution list.
        :param k: Number of individuals.
        """
        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(len(solutions))]

        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(len(solutions))]

        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solutions) + 1)]
        count = 0
        for p in range(len(solutions) - 1):
            for q in range(1, len(solutions)):
                dominance_test_result = self.preferences.compare(solutions[p], solutions[q])
                self.number_of_comparisons += 1
                if dominance_test_result in self.a_pref:
                    solutions[p].attributes['net_score'] = self.preferences.sigmaXY - self.preferences.sigmaYX
                    count += 1
                    ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif dominance_test_result in self.b_pref:
                    solutions[q].attributes['net_score'] = self.preferences.sigmaYX - self.preferences.sigmaXY
                    count += 1
                    ith_dominated[q].append(p)
                    dominating_ith[p] += 1

        self.ranked_sublists = []
        for i in range(len(solutions)):
            if dominating_ith[i] == 0 and 'net_score' in solutions[i].attributes.keys():
                solutions[i].attributes['dominance_ranking'] = 0
                front[0].append(solutions[i])

        if len(front[0]) > 0:
            self.ranked_sublists.append(front[0])
        return self.ranked_sublists


def clean_line(line: str) -> List[str]:
    line_ = line.replace('\"', "").split("//")
    line_ = [v.replace(',', ' ').replace('*', '') for v in line_[0].split()]
    rs = []
    for v in line_:
        rs += v.split()
    return rs
