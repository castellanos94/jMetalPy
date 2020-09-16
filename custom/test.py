import random
from typing import List

from custom.dtlz_problems import DTLZ1P
from custom.gd_problems import PortfolioSocialProblem, PortfolioSocialProblemGD, GDProblem
from custom.instance import PspInstance, DTLZInstance, PspIntervalInstance
from custom.interval import Interval
from custom.nsga3_c import NSGA3C
from custom.util_problem import InterClassNC, BestCompromise, ReferenceSetITHDM
from custom.utils import print_solutions_to_file, DIRECTORY_RESULTS, DMGenerator
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.core.problem import Problem
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


def dtlz_test(p: Problem, label: str = '', experiment: int = 50):
    # problem.reference_front = read_solutions(filename='resources/reference_front/DTLZ2.3D.pf')

    max_evaluations = 25000

    # references = ReferenceDirectionFromSolution(p)
    algorithm = NSGA3C(
        problem=p,
        population_size=100,
        reference_directions=UniformReferenceDirectionFactory(3, n_points=92),
        mutation=PolynomialMutation(probability=1.0 / p.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=30),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )
    bag = []
    for i in range(experiment):
        algorithm = NSGA3C(
            problem=p,
            population_size=92,
            reference_directions=UniformReferenceDirectionFactory(3, n_points=91),
            mutation=PolynomialMutation(probability=1.0 / p.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=30),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )
        progress_bar = ProgressBarObserver(max=max_evaluations)
        algorithm.observable.register(progress_bar)
        algorithm.run()
        print(len(algorithm.get_result()))
        bag = bag + algorithm.get_result()
    print(len(bag))
    print_solutions_to_file(bag, DIRECTORY_RESULTS + 'Solutions.bag._class_' + label + algorithm.label)
    ranking = FastNonDominatedRanking()

    ranking.compute_ranking(bag)
    front_ = ranking.get_subfront(0)
    class_fronts = [[], [], [], []]

    for s in front_:
        _class = problem.classifier.classify(s)
        if _class[0] > 0:
            class_fronts[0].append(s)
        elif _class[1] > 0:
            class_fronts[1].append(s)
        elif _class[2] > 0:
            class_fronts[2].append(s)
        else:
            class_fronts[3].append(s)
    print(len(class_fronts[0]), len(class_fronts[1]), len(class_fronts[2]), len(class_fronts[3]))
    front = []
    for f in class_fronts:
        if len(f) > 0:
            for s in f:
                front.append(s)
    print(len(front))
    # Save results to file
    print_solutions_to_file(class_fronts[0], DIRECTORY_RESULTS + 'front0._class_' + label + algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${p.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
    plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y', 'z'])
    plot_front.plot(class_fronts[0], label=label + 'F0 ' + algorithm.label,
                    filename=DIRECTORY_RESULTS + 'f0_class_' + label + algorithm.label,
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
    path = '/home/thinkpad/Documents/jemoa/src/main/resources/instances/gd/GD_ITHDM-UFCA.txt'
    path = '/home/thinkpad/Dropbox/GDM/GroupDecision_202001/_instances/GD_ITHDM-UFCC.txt'
    psp_instance.read_(path)

    problem = PortfolioSocialProblemGD(psp_instance)
    solutions = []
    for v in problem.instance_.attributes['best_compromise']:
        s = problem.create_from_string(v)
        problem.evaluate(s)
        solutions.append(s)
    for idx in range(100):
        s = problem.create_solution()
        problem.evaluate(s)
        solutions.append(s)
    classifier = InterClassNC(problem)
    for s in solutions:
        classifier.classify(s)


def looking_for_compromise(problem_: GDProblem):
    search = BestCompromise(problem, k=200)
    best, roi = search.make()
    print('Best compromise')
    print(best.variables, best.objectives, best.attributes['net_score'])
    print_solutions_to_file(roi, DIRECTORY_RESULTS + "compromise_" + problem_.get_name())


def reference_set(p: GDProblem):
    reference_set_outranking = ReferenceSetITHDM(p)
    reference_set_outranking.compute()


if __name__ == '__main__':
    random.seed(1)

    instance = DTLZInstance()
    path = '/home/thinkpad/PycharmProjects/jMetalPy/resources/DTLZ_INSTANCES/DTLZ1_Instance.txt'
    instance.read_(path)
    for k,v in instance.attributes.items():
        print(k,v)
    problem = DTLZ1P(instance)
    k = 5
    # dm_generator(3,7,7*[Interval(0,(9/8)*k*100)])
    # dtlz_test(problem, 'enfoque_fronts_')
    # reference_set(problem)
    # best = problem.generate_existing_solution(instance.attributes['best_compromise'][0])
    # print(best.objectives, best.constraints)
    #looking_for_compromise(problem)
