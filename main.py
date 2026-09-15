from GeneticAlgorithmInterface import VariableConstraintGA
from parameters import Parameters
from ProblemSpaceInterface import ProblemSpace
import numpy as np, copy

#DOESNT RESET THE WHOLE GRID WHEN NO BINS ARE DISCOVERED

class Solution():
    def __init__(self, id:int, bornInGen:int, solutionObject: object):
        self.id = id
        self.solutionObj = solutionObject
        self.bornIn = bornInGen
        self.fit = -1
        self.currentBin = -1
        self.age = 0
        self.valid = False

    def _update_fitness(self, problemSpace: object) -> float:
        """updates and returns solution fitness value"""
        self.fit = problemSpace.fitness(self.solutionObj)
        return self.fit
    
    def _update_current_bin(self, problemSpace: object) -> int:
        """updates and returns solutions bin index"""
        self.currentBin = problemSpace.place_in_bin(self.solutionObj)
        return self.currentBin

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
        self.solutionsNumber = 0
        self.currentGrid = self.build_random_grid()

    def _create_solution(self, solutionObject: object) -> Solution:
        """creates a solution - everytime a solution is created, it is evaluated, its bin assigned and check if its valid to get into quality bin"""
        self.solutionsNumber += 1
        solution = Solution(id=self.solutionsNumber, bornInGen=self.currentGen, solutionObject=solutionObject)
        solution._update_fitness(self.problem_space)
        solution._update_current_bin(self.problem_space)
        if self._check_valid(solution):
            self.put_in_bin_v0(solution) 
        return solution

    def _crossover(self, parent1:Solution, parent2:Solution) -> list[Solution]:
        """return two solutions, result of crossover between 2 other ones"""
        childrenSolutionObjs = self.problem_space.cross_over(parent1.solutionObj, parent2.solutionObj)
        output = []
        for childSolutionObj in childrenSolutionObjs:
            output.append(self._create_solution(childSolutionObj))
        return output

    def _mutation(self, toBeMutated:Solution, mutationRate:float, pos:tuple[int,int]=[-1,-1]) -> Solution:
        """return solution after applying mutation rate to it"""
        if self.parameter.useMutationGrid:
            mutationRate = self.parameter.mutationGrid[pos[1]][pos[0]]
        mutatedSolutionObj = self.problem_space.mutate(toBeMutated.solutionObj, mutationRate)
        return self._create_solution(mutatedSolutionObj)

    def build_random_grid(self) -> dict:
        """Returns a grid with random individuals obtained from the given problem space.
        Gridsize obtained from self.parameter"""
        grid = {}
        for y in range(len(self.parameter.grid)):
            grid[y] = {}
            for x in range(len(self.parameter.grid[y])):
                solution = self.problem_space.generate_random_individual()
                grid[y][x] = self._create_solution(solution)

        return grid

    def _check_valid(self, solution: Solution) -> bool:
        """returns true or false if a solution is valid. updates the parameter inside the solution too"""
        output = self.is_valid(solution.solutionObj)
        solution.valid = output
        return output

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

    def put_in_bin_v0(self, solution: Solution) -> None:
        """v0: Put a solution inside a bin if there's space and it performs better than the worst inside the bin
        This method doesn't verify the validity of solution."""
        targetBin = solution.currentBin
        self.qualityBins[targetBin].append(solution)
        self.qualityBins[targetBin] = sorted(self.qualityBins[targetBin], key=lambda solution: solution.fit, reverse=True)
        if (len(self.qualityBins[targetBin]) > self.maxBinSize):
            self.qualityBins[targetBin].pop()

    def reset_bins(self) -> None:
        self.oldQBins = copy.deepcopy(self.qualityBins)
        for binList in self.qualityBins:
            binList.clear()

        for binList in self.oldQBins:
            for solution in binList:
                solution._update_fitness(self.problem_space)
                solution._update_current_bin(self.problem_space)
                if self._check_valid(solution):
                    self.put_in_bin_v0(solution)

    def adapt(self) -> None:
        self.reset_bins()
        chosenOnes = []
        for binList in self.qualityBins:
            if len(binList) < 1: continue
            for solution in binList:
                chosenOnes.append(solution)

        if len(chosenOnes) < 1: 
            #no bins filled! So just re-evaluated current solutions.
            # self.currentGrid = self.build_random_grid() 
            for y in range(len(self.currentGrid)):
                for x in range(len(self.currentGrid[y])):
                    solution = self.currentGrid[y][x]
                    solution._update_fitness(self.problem_space)
                    solution._update_current_bin(self.problem_space)
                    if self._check_valid(solution):
                        self.put_in_bin_v0(solution)

        else:
            #the idea here is to fill the grid with solutions that are either mutations of good ones of children (crossover of them)
            newGrid = {}
            for y in range(len(self.currentGrid)):
                newGrid[y] = {}
                for x in range(len(self.currentGrid[y])):
                    parent1 = self.parameter.random.choice(chosenOnes)
                    candidates = [parent1]

                    if self.parameter.random.random() <= 0.5: #crossover parent1 with cell
                        parent2 = self.currentGrid[y][x]
                        children = self._crossover(parent1, parent2)
                        for child in children:
                            candidates.append(child)
                            if self._check_valid(child):
                                self.put_in_bin_v0(child) 

                    else: # no crossover, just mutate parent1
                        mutated = self._mutation(parent1, self.mutation_rate, (x,y))
                        candidates.append(mutated)
                        #check if its worth saving to bin
                        if self._check_valid(mutated):
                            self.put_in_bin_v0(mutated)

                    #the best stays in the grid
                    chosenOne = self.selectionFunc(self.qualityBins, candidates, {"tolerance":self.tolerance})
                    newGrid[y][x] = chosenOne
            self.currentGrid = newGrid

    def format_output(self) -> list[list[float, object]]:
        """formats output to competition requirements. formats from qualityBins"""
        output = []
        for qBin in self.qualityBins:
            thisBin = []
            for solution in qBin:
                thisBin.append((solution.fit, solution.solutionObj))
            output.append(thisBin)
        return output

    def set_up(self):
        """"Fills the currentGrid with random solutions from the problem space"""
        self.currentGrid = self.build_random_grid()

        for y in range(len(self.currentGrid)):
            for x in range(len(self.currentGrid[y])):
                solution = self.currentGrid[y][x]
                if self._check_valid(solution):
                    self.put_in_bin_v0(solution)

    def run_one_generation(self, made_change):
        """
        Complete a single generation of the algorithm

        Returns the population of valid (by both constant and variable constraints)
        individuals that are shorted in bins. Each individual should be stored as a tuple
        with the first value being the fitness and the second being the object

        EX: [[(fit1, obj1)], [], [(fit2, obj2), (fit3, obj3)], .... ]
        """
        self.currentGen += 1
        newGrid = {}
        if made_change:
            self.adapt()
        for y in range(len(self.currentGrid)):
            newGrid[y] = {}
            for x in range(len(self.currentGrid[y])):
                parent1 = self.currentGrid[y][x]
                candidates = [parent1]

                if self.parameter.random.random() <= self.cross_over_rate:
                    # crossover, need second parent -> mutates children
                    neighbors = self.get_neighbors(pos=(x, y), useToroid=self.parameter.toroidal)
                    parent2X, parent2Y = self.parentSelectionFunc(self.parameter.random, neighbors, {})
                    parent2 = self.currentGrid[parent2Y][parent2X]
                    children = self._crossover(parent1, parent2)
                    for child in children:
                        child = self._mutation(child, self.mutation_rate, (x,y))
                        candidates.append(child)
                        #check if its worth saving to bin
                        if self._check_valid(child):
                            self.put_in_bin_v0(child) 
                else:
                    # no crossover, just mutate
                    mutated = self._mutation(parent1, self.mutation_rate, (x,y)) 
                    candidates.append(mutated)
                    #check if its worth saving to bin
                    if self._check_valid(mutated):
                        self.put_in_bin_v0(mutated)

                #the best stays in the grid
                chosenOne = self.selectionFunc(self.qualityBins, candidates, {"tolerance":self.tolerance})
                newGrid[y][x] = chosenOne

        self.currentGrid = newGrid
        return self.format_output()