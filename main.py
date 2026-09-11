from GeneticAlgorithmInterface import VariableConstraintGA
from parameters import Parameters
from ProblemSpaceInterface import ProblemSpace
import numpy as np, copy

class YouAlgorithm(VariableConstraintGA):
    def __init__(self, parameters: Parameters, problem_space: ProblemSpace, number_generations, population_size, max_memory, cross_over_rate, mutation_rate, user, update_interval):
        super().__init__(problem_space, number_generations, population_size, max_memory, cross_over_rate, mutation_rate, user, update_interval)
        self.parameter = parameters
        self.selectionFunc = self.parameter.selectionFunc
        self.parentSelectionFunc = self.parameter.parentSelectionFunc
        self.tolerance = self.parameter.tolerance
        self.qualityBins = [[] for _ in range(self.problem_space.get_num_bins())]
        self.oldQBins = [[] for _ in range(self.problem_space.get_num_bins())]
        self.maxBinSize = 5
        self.currentGen = 0
        self.currentGrid = self.build_random_grid()

    def build_random_grid(self) -> dict:
        """Returns a grid with random individuals obtained from the given problem space.
        Gridsize obtained from self.parameter"""
        grid = {}
        for y in range(len(self.parameter.grid)):
            grid[y] = {}
            for x in range(len(self.parameter.grid[y])):
                solution = self.problem_space.generate_random_individual()
                fit = self.problem_space.fitness(solution)
                grid[y][x] = (fit, solution)
        return grid

    def get_neighbors(self, pos: tuple[int, int], useToroid: bool = False) -> list[tuple[int, int]]:
        """Returns a list of neighbors from given 'pos' considering toroidal or not
        Uses sizes of currentGrid. It'll break if grid is not initialized."""
        x, y = pos
        height = len(self.currentGrid)
        width = len(self.currentGrid[0]) if height > 0 else 0
        output = []

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue

                if useToroid:
                    neigh_x = (x + i) % width
                    neigh_y = (y + j) % height
                    output.append((neigh_x, neigh_y))
                else:
                    neigh_x = x + i
                    neigh_y = y + j
                    if 0 <= neigh_x < width and 0 <= neigh_y < height:
                        output.append((neigh_x, neigh_y))
        return output

    def put_in_bin_v0(self, fitness: float, solution: object) -> None:
        """v0: Put a solution inside a bin if there's space and it performs better than the worst inside the bin
        This method doesn't verify the validity of solution."""
        targetBin = self.problem_space.place_in_bin(solution)
        self.qualityBins[targetBin].append((fitness, solution))
        self.qualityBins[targetBin] = sorted(self.qualityBins[targetBin], key=lambda solution: solution[0], reverse=True)
        if (len(self.qualityBins[targetBin]) > self.maxBinSize):
            self.qualityBins[targetBin].pop()

    def reset_bins(self) -> None:
        self.oldQBins = copy.deepcopy(self.qualityBins)
        for binList in self.qualityBins:
            binList.clear()

        for binList in self.oldQBins:
            for _, solution in binList:
                if self.is_valid(solution):
                    fit = self.problem_space.fitness(solution)
                    if self.is_valid(solution):
                        self.put_in_bin_v0(fit, solution)

    def set_up(self):
        """"Fills the currentGrid with random solutions from the problem space"""
        self.currentGrid = self.build_random_grid()

        for y in range(len(self.currentGrid)):
            for x in range(len(self.currentGrid[y])):
                fit, solution = self.currentGrid[y][x]
                if self.is_valid(solution):
                    self.put_in_bin_v0(fit, solution)

    def run_one_generation(self, made_change):
        """
        Complete a single generation of the algorithm

        Returns the population of valid (by both constant and variable constraints)
        individuals that are shorted in bins. Each individual should be stored as a tuple
        with the first value being the fitness and the second being the object

        EX: [[(fit1, obj1)], [], [(fit2, obj2), (fit3, obj3)], .... ]
        """
        if made_change:
            self.reset_bins()
        self.currentGen += 1
        newGrid = {}
        for y in range(len(self.currentGrid)):
            newGrid[y] = {}
            for x in range(len(self.currentGrid[y])):
                chosenOne = (-1, -1)
                parent1Fit, parent1 = self.currentGrid[y][x]

                if self.parameter.random.random() <= self.cross_over_rate:
                    # crossover, need second parent -> mutates children
                    neighbors = self.get_neighbors(pos=(x, y), useToroid=self.parameter.toroidal)
                    parent2X, parent2Y = self.parentSelectionFunc(self.parameter.random, neighbors, {})
                    _, parent2 = self.currentGrid[parent2Y][parent2X]
                    children = self.problem_space.cross_over(parent1, parent2)
                    for child in children:
                        child = self.problem_space.mutate(child, self.mutation_rate)
                        fit = self.problem_space.fitness(child)
                        #check if its worth saving to bin
                        if self.is_valid(child):
                            self.put_in_bin_v0(fit, child)
                        # only best child goes to the grid
                        if fit >= chosenOne[0]:
                            chosenOne = (fit, child)
                else:
                    # no crossover, just mutate
                    mutated = self.problem_space.mutate(parent1, self.mutation_rate)
                    fit = self.problem_space.fitness(mutated)
                    chosenOne = (fit, mutated)
                    #check if its worth saving to bin
                    if self.is_valid(mutated):
                        self.put_in_bin_v0(fit, mutated)

                candidates = [(parent1Fit  , parent1     , int(self.problem_space.place_in_bin(parent1))),
                              (chosenOne[0], chosenOne[1], int(self.problem_space.place_in_bin(chosenOne[1])))]
                #the best stays in the grid
                chosenFit, chosenSolution = self.selectionFunc(self.qualityBins, candidates, {"tolerance":self.tolerance})
                newGrid[y][x] = (chosenFit, chosenSolution)

        self.currentGrid = newGrid
        return self.qualityBins