from abc import ABC
from typing import List, Generic, TypeVar

BitSet = List[bool]
S = TypeVar('S')


class Solution(Generic[S], ABC):
    """ Class representing solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int):
        self.number_of_objectives = number_of_objectives
        self.number_of_variables = number_of_variables

        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.variables = [[] for _ in range(self.number_of_variables)]
        self.attributes = {}

    def __eq__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False

    def __str__(self) -> str:
        return 'Solution(objectives={},variables={})'.format(self.objectives, self.variables)

    def is_feasible(self) -> bool:
        return (self.attributes.get('overall_constraint_violation') is None) or \
               (self.attributes['overall_constraint_violation'] == 0)


class BinarySolution(Solution[BitSet]):
    """ Class representing float solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int):
        super(BinarySolution, self).__init__(number_of_variables, number_of_objectives)

    def __copy__(self):
        new_solution = BinarySolution(
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution

    def get_total_number_of_bits(self) -> int:
        total = 0
        for var in self.variables:
            total += len(var)

        return total


class FloatSolution(Solution[float]):
    """ Class representing float solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int,
                 lower_bound: List[float], upper_bound: List[float]):
        super(FloatSolution, self).__init__(number_of_variables, number_of_objectives)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.lower_bound,
            self.upper_bound)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution


class IntegerSolution(Solution[int]):
    """ Class representing integer solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int,
                 lower_bound: List[int], upper_bound: List[int]):
        super(IntegerSolution, self).__init__(number_of_variables, number_of_objectives)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.lower_bound,
            self.upper_bound)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution


class PermutationSolution(Solution):
    """ Class representing permutation solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int):
        super(PermutationSolution, self).__init__(number_of_variables, number_of_objectives)

    def __copy__(self):
        new_solution = PermutationSolution(
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution
