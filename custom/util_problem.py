from typing import List, Tuple

import numpy as np

from custom.gd_problems import GDProblem
from custom.interval import Interval
from custom.utils import ITHDMPreferences, ITHDMPreferenceUF, ITHDMRanking, ITHDMDominanceComparator
from jmetal.algorithm.multiobjective.nsgaiii import ReferenceDirectionFactory
from jmetal.core.solution import Solution


class ReferenceDirectionFromSolution(ReferenceDirectionFactory):
    """
    Helper to create references points from initial solutions defined in the problem instance
    """

    def __init__(self, problem: GDProblem, normalize: bool = False):
        """
            :param problem: problem with instance associate
            :param normalize: objectives of initial solutions
         """
        super(ReferenceDirectionFromSolution, self).__init__(n_dim=problem.number_of_objectives)
        self.problem = problem
        self.instance = problem.instance_
        self.normalize = normalize

    def _compute(self):
        ref_dir = []
        for s in self.instance.initial_solutions:
            self.problem.evaluate(s)
            ref_dir.append(np.array(s.objectives))
        if self.normalize:
            min_f, max_f = np.min(ref_dir), np.max(ref_dir)
            norm = max_f - min_f
            ref_dir = (ref_dir - min_f) / norm

        return np.array(ref_dir)


class InterClassNC:
    """
    Required R2, R1 sets present in problem attributes.
    """

    def __init__(self, problem: GDProblem):
        self.problem = problem
        self.w = self.problem.create_solution()
        self.w.constraints = [0.0 for _ in range(self.problem.number_of_constraints)]

    def classify(self, x: Solution) -> List[int]:
        """
           Classify the x solution using the preference model associated with the dm.
           This result of this classification is a integer vector : [ HSAT, SAT, DIS, HDIS ]
        """
        categories = [0, 0, 0, 0]
        for dm in range(self.problem.instance_.attributes['dms']):
            if not self.problem.get_preference_model(dm).supports_utility_function:
                asc = self._asc_rule(x, dm)
                dsc = self._desc_rule(x, dm)
                if asc == dsc and asc != -1:
                    if asc < len(self.problem.instance_.attributes['r2'][dm]):
                        if self._is_high_sat(x, dm):
                            categories[0] += 1
                        else:
                            categories[1] += 1
                    else:
                        if self._is_high_dis(x, dm):
                            categories[3] += 1
                        else:
                            categories[2] += 1
                else:
                    categories[3] += 1
            else:
                is_sat = self._is_sat_uf(x, dm)
                is_high = self._is_high_sat_uf(x, dm)
                if is_sat and is_high:
                    categories[0] += 1
                elif not is_high and is_sat:
                    categories[1] += 1
                else:
                    is_dis = self._is_dis_uf(x, dm)
                    is_high_dis = self._is_high_dis_uf(x, dm)
                    if not is_high_dis and is_dis:
                        categories[2] += 1
                    else:
                        categories[3] += 1
        return categories

    def _is_sat_uf(self, x: Solution, dm: int) -> bool:
        """
            Def 18: DM is compatible with weighted-sum function model, The DM is said to be sat with a feasible S x iff
            the following conditions are fulfilled: i) For all w belonging to R1, x is alpha-preferred to w. ii) Theres is no
            z belonging to R2 such that z is alpha-preferred to x.
            """
        preference = ITHDMPreferenceUF(self.problem.objectives_type, self.problem.get_preference_model(dm))
        r1 = self.problem.instance_.attributes['r1'][dm]
        for idx in range(len(r1)):
            self.w.objectives = r1[idx]
            if preference.compare(x, self.w) > -1:
                return False
        r2 = self.problem.instance_.attributes['r2'][dm]
        count = 0
        for idx in range(len(r2)):
            self.w.objectives = r2[idx]
            if preference.compare(self.w, x) == -1:
                count += 1
        return bool(count == 0)

    def _is_dis_uf(self, x: Solution, dm: int) -> bool:
        """
           Def 19: DM is compatible with weighted-sum function model, The DM is said to be dissatisfied with a feasible S
           x if at least one of the following conditions is fulfilled: i) For all w belonging to R2, w is alpha-pref to x;
           ii) There is no z belonging to R1 such that x is alpha-pref to z.
           """

        preference = ITHDMPreferenceUF(self.problem.objectives_type, self.problem.get_preference_model(dm))
        r2 = self.problem.instance_.attributes['r2'][dm]
        count = 0
        for idx in range(len(r2)):
            self.w.objectives = r2[idx]
            if preference.compare(self.w, x) == -1:
                count += 1
        if count == len(r2):
            return True
        r1 = self.problem.instance_.attributes['r1'][dm]
        count = 0
        for idx in range(len(r1)):
            self.w.objectives = r1[idx]
            if preference.compare(x, self.w) == -1:
                count += 1
        return bool(count == 0)

    def _is_high_sat_uf(self, x: Solution, dm: int) -> bool:
        """
             Def 20: If the DM is sat with x, we say that the DM is high sat with x iff the following condition is also
             fulfilled: - For all w belonging to R2, x is alpha-pref to w.
            """
        preference = ITHDMPreferenceUF(self.problem.objectives_type, self.problem.get_preference_model(dm))
        r2 = self.problem.instance_.attributes['r2'][dm]
        for idx in range(len(r2)):
            self.w.objectives = r2[idx]
            if preference.compare(x, self.w) > -1:
                return False
        return True

    def _is_high_dis_uf(self, x: Solution, dm: int) -> bool:
        """
              Def 21: Suppose that the DM is dist with a S x, We say that the DM is highly dissatisfied with x if the
              following condition is also fulfilled - For all w belonging to R1, w is alpha-pref to x.
        """
        preference = ITHDMPreferenceUF(self.problem.objectives_type, self.problem.get_preference_model(dm))
        r1 = self.problem.instance_.attributes['r1'][dm]
        for idx in range(len(r1)):
            self.w.objectives = r1[idx]
            if preference.compare(self.w, x) > -1:
                return False
        return True

    def _is_high_sat(self, x: Solution, dm: int) -> bool:
        """
           The DM is highly satisfied with a satisfactory x if for each w in R2 we have xPr(Beta,Lambda)w
        """
        preferences = ITHDMPreferences(self.problem.objectives_type, self.problem.get_preference_model(dm))
        r2 = self.problem.instance_.attributes['r2'][dm]
        for idx in range(len(r2)):
            self.w.objectives = r2[idx]
            if preferences.compare(x, self.w) > -1:
                return False
        return True

    def _is_high_dis(self, x: Solution, dm: int) -> bool:
        """
            The DM is strongly dissatisfied with x if for each w in R1 we have wP(Beta, Lambda)x.
        """
        preferences = ITHDMPreferences(self.problem.objectives_type, self.problem.get_preference_model(dm))
        r1 = self.problem.instance_.attributes['r1'][dm]
        for idx in range(len(r1)):
            self.w.objectives = r1[idx]
            if preferences.compare(self.w, x) > -1:
                return False
        return True

    def _asc_rule(self, x: Solution, dm: int) -> int:
        preferences = ITHDMPreferences(self.problem.objectives_type, self.problem.get_preference_model(dm))
        r2 = self.problem.instance_.attributes['r2'][dm]
        category = -1
        for idx in range(len(r2)):
            self.w.objectives = r2[idx]
            v = preferences.compare(self.w, x)
            if v <= -1 or v == 2:
                category = idx
                break

        if category != -1:
            return category
        r1 = self.problem.instance_.attributes['r1'][dm]
        for idx in range(len(r1)):
            self.w.objectives = r1[idx]
            if preferences.compare(self.w, x) <= -1:
                category = idx
                break
        if category == -1:
            return category
        return category + len(r2)

    def _desc_rule(self, x: Solution, dm: int) -> int:
        preferences = ITHDMPreferences(self.problem.objectives_type, self.problem.get_preference_model(dm))
        category = -1
        r1 = self.problem.instance_.attributes['r1'][dm]
        for idx in range(len(r1)):
            self.w.objectives = r1[idx]
            if preferences.compare(x, self.w) <= -1:
                category = idx
                break

        if r1 != -1:
            category += len(r1)
        r2 = self.problem.instance_.attributes['r2'][dm]
        for idx in range(len(r2)):
            self.w.objectives = r2[idx]
            if preferences.compare(x, self.w) <= -1:
                category = idx
                break

        return category


class BestCompromise:
    """
    Looking for a best compromise in a solution sample using the dm preferences
    """

    def __init__(self, problem: GDProblem, sample_size=1000, dm: int = 0, k: int = 100000):
        self.problem = problem
        self.sample_size = sample_size
        self.dm = dm
        self.k = k
        self.preference = ITHDMPreferences(problem.objectives_type, problem.instance_.attributes['models'][dm])
        self.ranking = ITHDMRanking(self.preference, [-1], [-1])

    def make(self) -> Tuple[Solution, List[Solution]]:
        """returns candidate solutions
        Generates a sample of feasible solutions and compares them looking for an xPy
        or xSy relationship, finally orders the candidate solutions by crowding distance
        """
        bag = []
        while len(bag) < self.k:
            print('Check xPy inner :', len(bag))
            sample = [self.problem.generate_solution() for _ in range(self.sample_size)]
            candidates = self.ranking.compute_ranking(sample)
            if len(candidates) != 0:
                bag += candidates[0]
        print('Candidates size: ', len(bag))
        max_net_score = 0
        best_compromise = None
        for s in bag:
            if max_net_score < s.attributes['net_score']:
                max_net_score = s.attributes['net_score']
                best_compromise = s
        bag.remove(best_compromise)
        # Make ROI
        for x in bag:
            self.preference.compare(x, best_compromise)
            x.attributes['net_score'] = self.preference.sigmaXY

        roi = list(filter(lambda p: p.attributes['net_score'] >= self.preference.preference_model.beta, bag))
        return best_compromise, [best_compromise] + roi


class ReferenceSetITHDM:
    def __init__(self, problem_: GDProblem):
        """
        Conjuntos de references para InterClass-nC
        :NOTE: Solo para problemas de maximizacion, en caso de minimizacion se tendria que sumar para crear r1
        """
        self.problem_ = problem_
        self.instance_ = self.problem_.instance_

    @staticmethod
    def check_assumption44(bi: List[Interval], b: List[List[Interval]], pref, until: int) -> bool:
        for k in range(until):
            v = pref.compare_(bi, b[k])
            if v == -1 or v == -2:
                return False
        return True

    @staticmethod
    def check_assumption_74(w: List[Interval], z: List[List[Interval]], pref) -> bool:
        for idx in range(len(z)):
            v = pref.compare_(w, z[idx])
            if v != 1:
                return False
        return True

    def compute(self, is_objective: bool = False):
        dms = self.instance_.attributes['dms'] if 'dms' else 1 in self.instance_.attributes.keys()
        frontiers_objectives = self.instance_.attributes['frontiers']

        for dm in range(dms):
            best_compromise = self.problem_.generate_existing_solution(self.instance_.attributes['best_compromise'][dm],
                                                                       is_objective)
            print('best compromise', best_compromise.objectives)
            dif_ideal_front = []
            r2 = [[]]
            r1 = [[]]
            for v in frontiers_objectives:
                print(v)
            for idx in range(self.problem_.number_of_objectives):
                v = Interval(best_compromise.objectives[idx] - frontiers_objectives[dm][0][idx])
                dif_ideal_front.append(abs(v.midpoint()))
            print('DIF ', dif_ideal_front)
            dominance = ITHDMDominanceComparator(self.problem_.objectives_type,
                                                 self.problem_.get_preference_model(dm).alpha)
            if not self.problem_.get_preference_model(dm).supports_utility_function:
                # Step 1:
                for idx in range(self.problem_.number_of_objectives):
                    if self.problem_.objectives_type[idx]:
                        r2[0].append(Interval(best_compromise.objectives[idx] - dif_ideal_front[idx] / 3.0))
                    else:
                        r2[0].append(Interval(best_compromise.objectives[idx] + dif_ideal_front[idx] / 3.0))
                print('6 // OUTRANKING: R2(3) + R1(3)')
                print(str(r2[0]).replace('[', '').replace(']', ''))
                while dominance.dominance_test_(best_compromise.objectives, r2[0]) != -1:
                    val = dominance.dominance_test_(best_compromise.objectives, r2[0])
                    for idx, tmp in enumerate(r2[0]):
                        if self.problem_.objectives_type[idx]:
                            r2[0][idx] = tmp - dif_ideal_front[idx] / 3
                        else:
                            r2[0][idx] = tmp + dif_ideal_front[idx] / 3
                # Step 2: Creando r11 a partir de la frontera
                for idx in range(self.problem_.number_of_objectives):
                    if self.problem_.objectives_type[idx]:
                        r1[0].append((frontiers_objectives[dm][0][idx] - dif_ideal_front[idx] / 3))
                    else:
                        r1[0].append((frontiers_objectives[dm][0][idx] + dif_ideal_front[idx] / 3))
                while dominance.dominance_test_(r2[0], r1[0]) != -1:
                    for idx, tmp in enumerate(r1[0]):
                        if self.problem_.objectives_type[idx]:
                            r1[0][idx] = tmp - dif_ideal_front[idx] / 6
                        else:
                            r1[0][idx] = tmp + dif_ideal_front[idx] / 6
                # Step 3:  disminuir
                dif_r2_r1 = [abs((r2[0][idx] - r1[0][idx]).midpoint()) for idx in
                             range(self.problem_.number_of_objectives)]
                r2 = r2 + [
                    [v - dif_r2_r1[idx] / 4 if self.problem_.objectives_type[idx] else v + dif_r2_r1[idx] / 4 for idx, v
                     in enumerate(r2[0])]]
                pref = ITHDMPreferences(self.problem_.objectives_type, self.problem_.get_preference_model(dm))
                while dominance.dominance_test_(best_compromise.objectives,
                                                r2[1]) != -1 or not self.check_assumption44(r2[1], r2, pref, 1):
                    print(dominance.dominance_test_(best_compromise.objectives, r2[1]),
                          self.check_assumption44(r2[1], r2, pref, 1))
                    for idx in range(self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r2[1][idx] = r2[1][idx] - dif_r2_r1[idx] / 4
                        else:
                            r2[1][idx] = r2[1][idx] + dif_r2_r1[idx] / 4

                print(str(r2[1]).replace('[', '').replace(']', ''))
                # Step 3 r23 -> r22, r23[i] = r21 - r11/3
                r2 = r2 + [
                    [v - dif_r2_r1[idx] / 4 if self.problem_.objectives_type[idx] else v + dif_r2_r1[idx] / 4 for idx, v
                     in enumerate(r2[1])]]
                jdx = 0
                while dominance.dominance_test_(best_compromise.objectives,
                                                r2[2]) != -1 or not self.check_assumption44(r2[2], r2, pref, 2):
                    for idx in range(jdx, self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r2[2][idx] = r2[2][idx] - dif_r2_r1[idx] / 4
                        else:
                            r2[2][idx] = r2[2][idx] + dif_r2_r1[idx] / 4
                    jdx = jdx + 2 if jdx < self.problem_.number_of_objectives else 0

                print(str(r2[2]).replace('[', '').replace(']', ''))
                print(str(r1[0]).replace('[', '').replace(']', ''))
                dif_r2_r1 = [abs((r2[2][idx] - r1[0][idx]).midpoint()) for idx in
                             range(self.problem_.number_of_objectives)]
                r1 = r1 + [
                    [v - dif_r2_r1[idx] / 4 if self.problem_.objectives_type[idx] else v + dif_r2_r1[idx] / 4 for idx, v
                     in enumerate(r1[0])]]
                jdx = 0
                while dominance.dominance_test_(best_compromise.objectives,
                                                r1[1]) != -1 or not self.check_assumption44(r1[1], r1, pref, 1):
                    for idx in range(jdx, self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r1[1][idx] = r1[1][idx] - dif_r2_r1[idx] / 4
                        else:
                            r1[1][idx] = r1[1][idx] + dif_r2_r1[idx] / 4
                    jdx = jdx + 2 if jdx < self.problem_.number_of_objectives else 0
                print(str(r1[1]).replace('[', '').replace(']', ''))
                r1 = r1 + [
                    [v - dif_r2_r1[idx] / 4 if self.problem_.objectives_type[idx] else v + dif_r2_r1[idx] / 4 for idx, v
                     in enumerate(r1[1])]]
                jdx = 0
                while dominance.dominance_test_(best_compromise.objectives,
                                                r1[2]) != -1 or not self.check_assumption44(r1[2], r1, pref, 2):
                    for idx in range(jdx, self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r1[2][idx] = r1[2][idx] - dif_r2_r1[idx] / 4
                        else:
                            r1[2][idx] = r1[2][idx] + dif_r2_r1[idx] / 4
                    jdx = jdx + 2 if jdx < self.problem_.number_of_objectives else 0
                print(str(r1[2]).replace('[', '').replace(']', ''))
            else:
                # Step 1:
                for idx in range(self.problem_.number_of_objectives):
                    if self.problem_.objectives_type[idx]:
                        r2[0].append(Interval(best_compromise.objectives[idx] - dif_ideal_front[idx] / 2.0))
                    else:
                        r2[0].append(Interval(best_compromise.objectives[idx] + dif_ideal_front[idx] / 2.0))
                pref = ITHDMPreferenceUF(self.problem_.objectives_type, self.problem_.get_preference_model(dm))
                while dominance.dominance_test_(best_compromise.objectives, r2[0]) != -1:
                    for idx in range(self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r2[0][idx] -= dif_ideal_front[idx] / 3
                        else:
                            r2[0][idx] += dif_ideal_front[idx] / 3
                print('6 // UF: R2(3) + R1(3)')
                print(str(r2[0]).replace('[', '').replace(']', ''))
                r2 += [[v - dif_ideal_front[idx] / 3 if self.problem_.objectives_type[idx] else v + dif_ideal_front[
                    idx] / 3 for idx, v in enumerate(r2[0])]]
                while dominance.dominance_test_(best_compromise.objectives, r2[0]) != -1 and pref.compare_(r2[1],
                                                                                                           r2[0]) != 0:
                    for idx in range(self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r2[1][idx] -= dif_ideal_front[idx] / 3
                        else:
                            r2[1][idx] += dif_ideal_front[idx] / 3

                print(str(r2[1]).replace('[', '').replace(']', ''))
                r2 += [[v - dif_ideal_front[idx] / 3 if self.problem_.objectives_type[idx] else v + dif_ideal_front[
                    idx] / 3 for idx, v in enumerate(r2[1])]]
                while dominance.dominance_test_(best_compromise.objectives, r2[1]) != -1 and \
                        pref.compare_(r2[2], r2[0]) != 0 and pref.compare_(r2[2], r2[1]) != 0:
                    for idx in range(self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r2[1][idx] -= dif_ideal_front[idx] / 3
                        else:
                            r2[1][idx] += dif_ideal_front[idx] / 3

                print(str(r2[2]).replace('[', '').replace(']', ''))
                r1[0] = [v - dif_ideal_front[idx] if self.problem_.objectives_type[idx] else v + dif_ideal_front[idx]
                         for idx, v in enumerate(r2[2])]
                while not self.check_assumption_74(r1[0], r2, pref):
                    for idx in range(self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r1[0] -= dif_ideal_front[idx] / 3
                        else:
                            r1[0] += dif_ideal_front[idx] / 3
                print(str(r1[0]).replace('[', '').replace(']', ''))
                r1 += [[v - dif_ideal_front[idx] / 3 if self.problem_.objectives_type[idx] else v + dif_ideal_front[
                    idx] / 3 for idx, v in enumerate(r1[0])]]
                while not self.check_assumption_74(r1[1], r2, pref) and pref.compare_(r1[0], r1[1]) != 0:
                    for idx in range(self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r1[1] -= dif_ideal_front[idx] / 3
                        else:
                            r1[1] += dif_ideal_front[idx] / 3
                print(str(r1[1]).replace('[', '').replace(']', ''))
                r1 += [[v - dif_ideal_front[idx] / 3 if self.problem_.objectives_type[idx] else v + dif_ideal_front[
                    idx] / 3 for idx, v in enumerate(r1[1])]]
                while not self.check_assumption_74(r1[2], r2, pref) and pref.compare_(r1[0], r1[2]) != 0 and \
                        pref.compare_(r1[1], r1[2]) != 0:
                    for idx in range(self.problem_.number_of_objectives):
                        if self.problem_.objectives_type[idx]:
                            r1[2] -= dif_ideal_front[idx] / 3
                        else:
                            r1[2] += dif_ideal_front[idx] / 3
                print(str(r1[2]).replace('[', '').replace(']', ''))
