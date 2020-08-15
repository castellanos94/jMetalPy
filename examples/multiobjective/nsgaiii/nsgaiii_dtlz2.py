import random

from custom.utils import DIRECTORY_RESULTS, print_solutions_to_file
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.lab.visualization import Plot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import DTLZ1
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.solution import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    random.seed(8435)

    problem = DTLZ1()
    problem.reference_front = read_solutions(filename='resources/reference_front/DTLZ2.3D.pf')

    max_evaluations = 25000
    experiment = 50
    bag = []
    algorithm = NSGAIII(
        problem=problem,
        population_size=92,
        reference_directions=UniformReferenceDirectionFactory(3, n_points=91),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=30),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )
    for i in range(experiment):
        algorithm = NSGAIII(
            problem=problem,
            population_size=92,
            reference_directions=UniformReferenceDirectionFactory(3, n_points=91),
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=30),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )
        progress_bar = ProgressBarObserver(max=max_evaluations)
        algorithm.observable.register(progress_bar)

        algorithm.run()
        bag += algorithm.get_result()
    print(len(bag))
    print_solutions_to_file(bag, DIRECTORY_RESULTS + 'Solutions.bag.' + algorithm.label)
    ranking = FastNonDominatedRanking()

    ranking.compute_ranking(bag)
    front = ranking.get_subfront(0)
    print(len(front))
    # Save results to file
    print_solutions_to_file(front, DIRECTORY_RESULTS + 'front0.' + algorithm.label)
    plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y', 'z'])
    plot_front.plot(front, label='NSGAII-ZDT1', filename=DIRECTORY_RESULTS + 'NSGAII-ZDT1', format='png')
    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
