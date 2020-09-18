from typing import List

from custom.dtlz_problems import DTLZ1P
from custom.gd_problems import PortfolioSocialProblem, PortfolioSocialProblemGD, GDProblem, FloatProblemGD
from custom.instance import PspInstance, DTLZInstance, PspIntervalInstance
from custom.interval import Interval
from custom.nsga3_c import NSGA3C
from custom.util_problem import InterClassNC, BestCompromise, ReferenceSetITHDM
from custom.utils import print_solutions_to_file, DIRECTORY_RESULTS, DMGenerator, clean_line, ITHDMRanking, \
    ITHDMPreferences
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
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


def dtlz_test(p: FloatProblemGD, label: str = '', experiment: int = 30):
    # problem.reference_front = read_solutions(filename='resources/reference_front/DTLZ2.3D.pf')

    max_evaluations = 25000

    # references = ReferenceDirectionFromSolution(p)
    algorithm = NSGA3C(
        problem=p,
        population_size=100,
        reference_directions=UniformReferenceDirectionFactory(p.instance_.n_obj, n_points=92),
        mutation=PolynomialMutation(probability=1.0 / p.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=30),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )
    bag = []
    total_time = 0
    for i in range(experiment):
        algorithm = NSGA3C(
            problem=p,
            population_size=92,
            reference_directions=UniformReferenceDirectionFactory(p.instance_.n_obj, n_points=91),
            mutation=PolynomialMutation(probability=1.0 / p.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=30),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )
        progress_bar = ProgressBarObserver(max=max_evaluations)
        algorithm.observable.register(progress_bar)
        algorithm.run()
        total_time += algorithm.total_computing_time
        bag = bag + algorithm.get_result()
    print(len(bag))
    print('Total computing time:', total_time)
    print('Average time: ', str(total_time / experiment))
    print_solutions_to_file(bag, DIRECTORY_RESULTS + 'Solutions.bag._class_' + label + algorithm.label)
    ranking = FastNonDominatedRanking()

    ranking.compute_ranking(bag)
    front_ = ranking.get_subfront(0)
    plot_front = Plot(title='Pareto front approximation' + ' Sin desplazamiento de frente')
    plot_front.plot(front_, label=label + 'F0 ' + algorithm.label,
                    filename=DIRECTORY_RESULTS + 'F0_class_' + 'original_' + algorithm.label,
                    format='png')
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
    plot_front = Plot(title='Pareto front approximation' + label)
    plot_front.plot(class_fronts[0], label=label + 'F0 ' + algorithm.label,
                    filename=DIRECTORY_RESULTS + 'F0_class_' + label + algorithm.label,
                    format='png')


def dm_generator(number_of_objectives: int, number_of_variables: int, max_objectives: List[Interval]):
    print(number_of_variables, number_of_objectives, max_objectives)
    generator = DMGenerator(number_of_variables=number_of_variables, number_of_objectives=number_of_objectives,
                            max_objectives=max_objectives)
    w, veto = generator.make()
    print(number_of_objectives)
    print(number_of_variables)
    print(', '.join(map(str, w)))
    print(', '.join(map(str, veto)))
    print(Interval(0.51, 0.75))
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


def read_roi_from_obj(problem: GDProblem, roi_file: str):
    with open(roi_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    print(len(content))


def loadDataWithClass(p: DTLZ1P, bag_path: str, label: str):
    with open(bag_path) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    _bag = []
    for element in content:
        if not ('variables * objectives * constraints' in element):
            line = clean_line(element)
            _bag.append(p.generate_existing_solution([float(_var) for _var in line[:p.number_of_variables]]))
    print(len(_bag))
    # classifier solutions
    _class = [[], [], [], []]
    for _solution in _bag:
        _class_vector = p.classifier.classify(_solution)
        if _class_vector[0] > 0:
            _class[0].append(_solution)
        elif _class_vector[1] > 0:
            _class[1].append(_solution)
        elif _class_vector[2] > 0:
            _class[2].append(_solution)
        else:
            _class[3].append(_solution)
    print(len(_class[0]), len(_class[1]), len(_class[2]), len(_class[3]))
    _front = None
    idx = 0
    for i, _f in enumerate(_class):
        if len(_f) > 0:
            _front = _f
            idx = i
            break
    print(idx, len(_front))
    # Save results to file
    print_solutions_to_file(_front, DIRECTORY_RESULTS + 'front0._class_' + label + p.get_name())
    alabels = []
    for obj in range(p.number_of_objectives):
        alabels.append('Obj-' + str(obj))
    plot_front = Plot(title='Pareto front approximation ' + label + 'F' + str(idx) + p.get_name(), axis_labels=alabels)
    plot_front.plot(_front, label=label + 'F' + str(idx) + p.get_name(),
                    filename=DIRECTORY_RESULTS + 'F0_class_' + label + p.get_name(),
                    format='png')
    # Show best compromise
    _best = p.generate_existing_solution(p.instance_.attributes['best_compromise'][0])
    print('Best compromise:', _best.objectives)


def load_objectives_from_gdm_file(p: DTLZ1P, obj_path: str):
    with open(obj_path) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    _bag = []
    for element in content:
        line = clean_line(element)
        solution_ = p.generate_solution()
        solution_.objectives = [float(_var) for _var in line[:p.number_of_objectives]]
        _bag.append(solution_)
        print(solution_.objectives)
    print(len(_bag))
    preference = ITHDMPreferences(p.objectives_type, p.instance_.attributes['models'][0])
    ranking = ITHDMRanking(preference, [-1], [-1])
    candidates = ranking.compute_ranking(_bag)
    print('Candidates size: ', len(candidates))
    max_net_score = 0
    best_compromise = None
    for s in candidates:
        if max_net_score < s.attributes['net_score']:
            max_net_score = s.attributes['net_score']
            best_compromise = s
    candidates.remove(best_compromise)
    # Make ROI
    for x in candidates:
        candidates.preference.compare(x, best_compromise)
        x.attributes['net_score'] = preference.sigmaXY

    roi = list(filter(lambda p: p.attributes['net_score'] >= preference.preference_model.beta, candidates))
    print('Best compromise')
    print(best_compromise.objectives)
    print('ROI', len(roi))
    for s in roi:
        print(s.objectives)


if __name__ == '__main__':
    # random.seed(145600)

    instance = DTLZInstance()
    path = '/home/thinkpad/PycharmProjects/jMetalPy/resources/DTLZ_INSTANCES/DTLZ1_Instance.txt'
    path = '/home/thinkpad/PycharmProjects/jMetalPy/resources/DTLZ_INSTANCES/DTLZ1P_10.txt'
    instance.read_(path)

    problem = DTLZ1P(instance)
    for k, v in instance.__dict__.items():
        print(k, v)

    k = 5
    obj = 3
    # loadDataWithClass(problem,
    #                 '/home/thinkpad/PycharmProjects/jMetalPy/results/Solutions.bag._class_enfoque_fronts_NSGAIII_custom.DTLZ1P_10.csv',
    #                 'enfoque_fronts_')
    load_objectives_from_gdm_file(problem,
                                  '/home/thinkpad/PycharmProjects/jMetalPy/resources/DTLZ_INSTANCES/objetivos_nelson.csv')
    # dm_generator(obj, 14, obj * [Interval(0, (9 / 8) * k * 100)])
    # dtlz_test(problem, 'enfoque_fronts_')
    # reference_set(problem)
    # best = problem.generate_existing_solution(instance.attributes['best_compromise'][0])
    # print(best.objectives, best.constraints)
    # looking_for_compromise(problem)
