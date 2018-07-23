# Reference: https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py

import array, random, json
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from deap import algorithms, base, benchmarks, creator, tools
from deap.benchmarks.tools import diversity, convergence, hypervolume

# init

# 遺伝子の最小値と最大値(今回は0~1)
BOUND_LOW, BOUND_UP = 0.0, 1.0
# geneの数: individualの中の数
NDIM = 30

# 問題定義:適応度と個体
# Fitnessの最小値の定義．最小化なので負の値を渡す
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
# 遺伝子(Individual)の定義
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def init(seed=None):
    # 決まった乱数を出すためにseedは固定する
    random.seed(seed)
    # 世代数，個体突然変異率，交叉率
    NGEN, MUTPB, CXPB = 250, 100, 0.9

    # 統計
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Logging Data
    logbook = tools.Logbook()
    # specify attribute
    logbook.header = "gen", "evals", "min", "max"

    # 初期世代の作成．世代の個数はMUTPBで指定(100個)
    pop = toolbox.population(n=MUTPB)
    # 初期世代を格納
    # pop_ini = pop[:]

    # Evaluate the individuals with an invalid fitness
    # invalidなindividualを保存
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 選択
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # 世代間プロセスを開始
    for gen in range(1, NGEN):
        # 次世代の個体を選択
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # 偶数番目と奇数番目を取り出す
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                # crossover
                toolbox.mate(ind1, ind2)

            # 突然変異
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # individualの評価
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 次世代の選択
        pop = toolbox.select(pop + offspring, MUTPB)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is {}".format(hypervolume(pop, [11.0, 11.0])))

    return pop, logbook

def main():
    # pop, stats = init(64)


    with open("pareto_front/zdt1_front.json") as optimal_front_data:
        optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

    pop, stats = init()
    pop.sort(key=lambda x: x.fitness.values)

    print(stats)
    print("Convergence: ", convergence(pop, optimal_front))
    print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    # import matplotlib.pyplot as plt
    # import numpy

    front = np.array([ind.fitness.values for ind in pop])
    optimal_front = np.array(optimal_front)
    plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()

if __name__ == '__main__':
    main()
