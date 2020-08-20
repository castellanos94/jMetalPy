from typing import List

import numpy as np

from custom.gd_problems import GDProblem
from custom.utils import ITHDMPreferences, ITHDMPreferenceUF, ITHDMRanking
from jmetal.algorithm.multiobjective.nsgaiii import ReferenceDirectionFactory
from jmetal.core.solution import Solution
from jmetal.util.density_estimator import CrowdingDistance


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
        print(ref_dir)
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
                if asc == dsc:
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
            if preferences.compare(self.w, x) <= -1:
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
        if category != -1:
            return category + len(r1)
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

    def __init__(self, problem: GDProblem, sample_size=1000, dm: int = 0, k: int = 20):
        self.problem = problem
        self.sample_size = sample_size
        self.dm = dm
        self.k = k
        self.preference = ITHDMPreferences(problem.objectives_type, problem.instance_.attributes['models'][dm])
        self.ranking = ITHDMRanking(self.preference, [-2, -1], [2])

    def make(self) -> List[Solution]:
        """returns candidate solutions
            Generates a sample of feasible solutions and compares them looking for an xPy or xSy relationship, finally orders the
            candidate solutions by crowding distance
        """
        bag = []
        while len(bag) < self.k:
            print('Check xPy inner :', len(bag))
            sample = [self.problem.generate_solution() for _ in range(self.sample_size)]
            candidates = self.ranking.compute_ranking(sample)
            if len(candidates) != 0:
                bag += candidates[0]
        print('Candidates size: ', len(bag))
        CrowdingDistance().compute_density_estimator(bag)
        return bag
