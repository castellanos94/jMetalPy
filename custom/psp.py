import random

from custom.instance import PspInstance
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.util.comparator import DominanceComparator
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solution import print_solution_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations


class PortfolioSocialProblem(BinaryProblem):
    def __init__(self, instance: PspInstance):
        super(PortfolioSocialProblem, self).__init__()
        self.number_of_bits = instance.n_var
        self.number_of_variables = 1
        self.number_of_objectives = instance.n_obj
        self.number_of_constraints = instance.n_constraints
        self.instance = instance
        self.budget = instance.budget

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(
                self.number_of_bits)]

        return new_solution

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        budget = 0
        objectives = [0 for v in range(self.number_of_objectives)]

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                budget += self.instance.projects[index][0]
                for obj in range(0, self.number_of_objectives):
                    objectives[obj] += self.instance.projects[index][obj + 3]
        solution.objectives = [-1 * obj for obj in objectives]

        solution.constraints = [self.budget - budget]
        return solution

    def get_name(self) -> str:
        return 'PortfolioSocialProblem'


if __name__ == '__main__':
    path = '/home/thinkpad/Documents/jemoa/src/main/resources/instances/psp/o4p25_0.txt'
    pspInstance = PspInstance()
    pspInstance.read_(path)
    max_evaluations = 20000
    binary_string_length = pspInstance.n_var
    problem = PortfolioSocialProblem(pspInstance)

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=1,
        mutation=BitFlipMutation(probability=1.0 / binary_string_length),
        crossover=SPXCrossover(probability=1.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=DominanceComparator()
    )
    progress_bar = ProgressBarObserver(max=max_evaluations)
    algorithm.observable.register(progress_bar)

    algorithm.run()
    front = algorithm.get_result()
    for s in front:
        s.objectives = [-1 * obj for obj in s.objectives]
    # Save results to file
    # print_function_values_to_file(front, 'FUN.' + algorithm.get_name() + "-" + problem.get_name())
    print_solution_to_file(front, 'SOLUTIONS.' + algorithm.get_name() + "-" + problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
