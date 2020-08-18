import random
from typing import List

from custom.gd_problems import PortfolioSocialProblem, DTLZ1Preferences, PortfolioSocialProblemGD
from custom.instance import PspInstance, DTLZInstance, PspIntervalInstance
from custom.interval import Interval
from custom.util_problem import ReferenceDirectionFromSolution, InterClassNC
from custom.utils import print_solutions_to_file, DIRECTORY_RESULTS, DMGenerator
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.lab.visualization import Plot
from jmetal.operator import BitFlipMutation, SPXCrossover, PolynomialMutation, SBXCrossover
from jmetal.util.comparator import DominanceComparator
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import StoppingByEvaluations


def psp_test():
    path = '/home/thinkpad/Documents/jemoa/src/main/resources/instances/psp/o4p25_0.txt'
    psp_instance = PspInstance()
    psp_instance.read_(path)
    max_evaluations = 50000
    binary_string_length = psp_instance.n_var
    problem = PortfolioSocialProblem(psp_instance)

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
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
    print_solutions_to_file(front, DIRECTORY_RESULTS + 'SOLUTIONS.' + algorithm.get_name() + "-" + problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))


def dtlz1_test():
    instance = DTLZInstance()
    path = '/home/thinkpad/Documents/jemoa/src/main/resources/instances/dtlz/DTLZInstance.txt'
    instance.read_(path)
    problem = DTLZ1Preferences(instance)

    # problem.reference_front = read_solutions(filename='resources/reference_front/DTLZ2.3D.pf')

    max_evaluations = 25000
    experiment = 30
    algorithm = NSGAIII(
        problem=problem,
        population_size=100,
        reference_directions=ReferenceDirectionFromSolution(problem),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=30),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )
    bag = []
    for i in range(experiment):
        algorithm = NSGAIII(
            problem=problem,
            population_size=92,
            reference_directions=ReferenceDirectionFromSolution(problem),
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=30),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )
        progress_bar = ProgressBarObserver(max=max_evaluations)
        algorithm.observable.register(progress_bar)
        algorithm.run()

        bag = bag + algorithm.get_result()
    print(len(bag))
    print_solutions_to_file(bag, DIRECTORY_RESULTS + 'Solutions.bag.' + algorithm.label)
    ranking = FastNonDominatedRanking()

    ranking.compute_ranking(bag)
    front = ranking.get_subfront(0)
    print(len(front))
    # Save results to file
    print_solutions_to_file(front, DIRECTORY_RESULTS + 'front0.' + algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
    plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y', 'z'])
    plot_front.plot(front, label='NSGAII-ZDT1-preferences_bag', filename=DIRECTORY_RESULTS + 'NSGAII-ZDT1-p_f0',
                    format='png')


def dm_generator(number_of_objectives: int, number_of_variables: int, max_objectives: List[Interval]):
    print(number_of_variables, number_of_objectives, max_objectives)
    generator = DMGenerator(number_of_variables=number_of_variables, number_of_objectives=number_of_objectives,
                            max_objectives=max_objectives)
    w, v = generator.make()
    print(number_of_objectives)
    print(number_of_variables)
    print(w)
    print(v)
    print(Interval(0.51, 0.67))


def test_classifier():
    psp_instance = PspIntervalInstance()
    psp_instance.read_('/home/thinkpad/Documents/jemoa/src/main/resources/instances/gd/GD_ITHDM-UFCA.txt')

    problem = PortfolioSocialProblemGD(psp_instance)
    solutions = []
    for v in problem.instance_.attributes['best_compromise']:
        s = problem.create_from_string(v)
        problem.evaluate(s)
        solutions.append(s)
    for idx in range(10):
        s = problem.create_solution()
        problem.evaluate(s)
        solutions.append(s)
    classifier = InterClassNC(problem)
    for s in solutions:
        print(s.constraints, s.budget, classifier.classify(s))


if __name__ == '__main__':
    random.seed(1)
    # dm_generator(4, 7, 4 * [Interval(0, 0.5)])
    test_classifier()
