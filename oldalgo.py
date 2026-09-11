from GeneticAlgorithmInterface import VariableConstraintGA 
from parameters import Parameters
import numpy as np, random as rd, time, os, shutil
from datetime import datetime
from multiprocessing import Pool
####
from ProblemSpaceInterface import ProblemSpace 
from ProblemSpaces.LodeRunner.LodeRunnerProblemSpace import LodRunnerProblemSpace
from ProblemSpaces.LogicPuzzles.LogicPuzzleSpace import LogicPuzzleSpace
# from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
from Personas.Exploratory import ExploratoryUser

class YouAlgorithm(VariableConstraintGA):
    def __init__(self, parameters:Parameters, problem_space: ProblemSpace, number_generations, population_size, max_memory, cross_over_rate, mutation_rate, user, update_interval, infeasible_rate = 0.5, elitism = 0.3, height=4, advice_thres = 0.5, advice_height=2):
        super().__init__(problem_space, number_generations, population_size, max_memory, cross_over_rate, mutation_rate, user, update_interval)
        self.parameter = parameters
        self.currentGrid = self.build_random_grid()
        self.qualityBins = [[] for _ in range(self.problem_space.get_num_bins())]
        self.maxBinSize = 10#int(self.max_memory / self.problem_space.get_num_bins())
        self.currentGen = 0

        ##### ERASE THESE LATER
        self.execFolder = self.parameter.execFolder
        shutil.copy("parameters.py", self.execFolder)
        self.outputCSV = f"{self.execFolder}/data.csv"
        with open(self.outputCSV, mode='a', encoding='utf-8') as f:
            f.write("generation, x, y, fitness\n")
        self.binsCSV = f"{self.execFolder}/bins.csv"
        with open(self.binsCSV, mode='a', encoding='utf-8') as f:
            f.write("gen, bins\n")

    def print_bins(self) -> None:
        for i, bin in enumerate(self.qualityBins):
            print(f"Bin {i} - {bin}")

    def print_grid(self) -> None:
        for y in self.currentGrid:
            for x in self.currentGrid[y]:
                print(f"({x},{y}) - {self.currentGrid[y][x]}")

    def build_random_grid(self) -> dict:
        """Returns a grid with random individuals obtained from the given problem space.
        Gridsize obtained from self.parameter"""
        grid = {}
        for y in range(len(self.parameter.grid)):
            grid[y] = {}
            for x in range(len(self.parameter.grid[y])):
                grid[y][x] = 0
                solution = self.problem_space.generate_random_individual()
                fit      = self.problem_space.fitness(solution)
                grid[y][x] = (fit, solution)
        return grid

    def get_neighbors(self, pos: tuple[int, int], useToroid: bool = False) -> list[tuple[int, int]]:
        """Returns a list of neighbors from given 'pos' considering toroidal or not
        Uses sizes of currentGrid. It'll brake if grid is not initialized."""
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

    def put_in_bin_v0(self, fitness:float, solution: object) -> None:
        """v0: Put a solution inside a bin if theres is space and it performs better than the worst inside the bin
        This method doesnt verify the vality of solution :)"""
        targetBin = self.problem_space.place_in_bin(solution)
        self.qualityBins[targetBin].append((fitness, solution))
        self.qualityBins[targetBin] = sorted(self.qualityBins[targetBin], key=lambda solution: solution[0], reverse=True)    
        if (len(self.qualityBins[targetBin]) > self.maxBinSize):
            self.qualityBins[targetBin].pop()            

    def select_parent2_random(self, neighbors:list[tuple[int, int]]) -> tuple[int, int]:
        """Returns a random neighbor position from the list"""
        return self.parameter.random.choice(neighbors)

    def check_constraints(self, solution: object) -> bool:
        """Returns true if solution satisfied all current constraints"""
        #I could count the amount of satisfied constraints and use it...
        for con in self.problem_space.get_constant_constraints():
            if con.apply(solution): continue
            else: return False
        return True

    def set_up(self): 
        """
        Insert all your set up code here 

        You can generation an initial population 
        of individuals. However you may only 
        generate self.population_size in this method, and 
        can only store up to self.max_memory individuals  
        in total   
        
        We provide the useful functions 
        and values available for you to use here 
        """
        self.currentGrid = self.build_random_grid() 

        for y in range(len(self.currentGrid)):
            for x in range(len(self.currentGrid[y])):
                fit, solution = self.currentGrid[y][x]
                if self.check_constraints(solution):
                    self.put_in_bin_v0(fit, solution)   

        self.save_current_grid(0)  

        # self.population_size # the max number of individuals you can generate per generation 
        # self.max_memory # the max number of individuals you can store at any time (always > then pop size)
        # self.mutation_rate # the rate of mutation to give mutation function 
        # self.cross_over_rate # the ratio of time you should preform the cross over function 
        # self.variable_constraints # the current list of variable constraints 

        # ind1 = self.problem_space.generate_random_individual() # randomly generate a new individual 
        # ind2 = self.problem_space.generate_random_individual() 

        # fit = self.problem_space.fitness(ind1) # quality value of individual 
        # ind3 = self.problem_space.mutate(ind1, self.mutation_rate) # preform mutation 
        # child1, child2 = self.problem_space.cross_over(ind1, ind2) # preform cross over 

        # cons = self.problem_space.get_constant_constraints() # list of static constraints 
 
        # print(f"bin number {self.problem_space.get_num_bins()}") # number of diversity bins in problem space 
        # print(f"which bin {self.problem_space.place_in_bin(ind1)}") # get the index of bin ind should be placed in 
        
        # you can check if individuals satisfy a constant through the apply function 
        # cons[0].apply(ind1) # returns true if constraint is satisfied 

    def run_one_generation(self, made_change): 
        """
        Complete a single generation of the algorithm

        Returns the population of valid (by both constant and variable constraints)
        individuals that are shorted in bins. Each individual should be stored as a tuple
        with the first value being the fitness and the second being the object 

        EX: [[(fit1, obj1)], [], [(fit2, obj2), (fit3, obj3)], .... ] 
        
        """
        self.currentGen += 1
        started = time.perf_counter()
        newGrid = {}
        fitnessList = []
        for y in range(len(self.currentGrid)):
            newGrid[y] = {}
            for x in range(len(self.currentGrid[y])):
                chosenOne = (-1, -1)
                parent1Fit, parent1 = self.currentGrid[y][x]

                if self.parameter.random.random() <= self.cross_over_rate:
                    #crossover, need second parent -> mutates children
                    neighbors = self.get_neighbors(pos=(x,y), useToroid=self.parameter.toroidal)
                    parent2X, parent2Y = self.select_parent2_random(neighbors=neighbors)
                    parent2Fit, parent2 = self.currentGrid[parent2Y][parent2X]
                    children = self.problem_space.cross_over(parent1, parent2)
                    for child in children:
                        child = self.problem_space.mutate(child, self.mutation_rate)
                        fit   = self.problem_space.fitness(child)  
                        #check if its worth on saving to bin
                        if self.check_constraints(child):
                            self.put_in_bin_v0(fit, child)

                        #only best child goes to the grid
                        if fit >= chosenOne[0]:
                            chosenOne = (fit, child)
                else:
                    #no crossover, just mutate
                    chosenOne = (parent1Fit, parent1)
                    mutated = self.problem_space.mutate(chosenOne[1], self.mutation_rate)
                    fit     = self.problem_space.fitness(mutated)
                    chosenOne = (fit, mutated)
                    #check if its worth on saving to bin
                    if self.check_constraints(mutated):
                        self.put_in_bin_v0(fit, mutated)

                if chosenOne[0] >= parent1Fit:
                    newGrid[y][x] = chosenOne
                else:
                    newGrid[y][x] = (parent1Fit, parent1)

        self.currentGrid = newGrid
        finished = time.perf_counter()
        if (finished-started) > 30:
            print(f"Warning! Generation {self.currentGen} finished in {finished-started}")
        self.save_current_grid(self.currentGen)
        self.save_bins_state(self.currentGen)
        return self.qualityBins

    def save_current_grid(self, gen:int) -> None:
        with open(self.outputCSV, mode='a', encoding='utf-8') as f:
            for y in self.currentGrid:
                for x in self.currentGrid[y]:
                    fit, solution = self.currentGrid[y][x]
                    f.write(f"{gen}, {x},{y}, {fit}\n")

    def save_bins_state(self, gen:int) -> None:
        output = f"\nGeneration: {gen},"
        for i, bin in enumerate(self.qualityBins):
            output += f"\n--Bin{i}:"
            for fit, _ in bin:
                output += f" {fit},"
        with open(self.binsCSV, mode='a', encoding='utf-8') as f:
            f.write(output)


def exec_wrapper(args: tuple) -> dict:
    import random
    import numpy as np
    expFolder, seed = args
    start_time = time.time()

    try:
        #problem space
        problem_space = LogicPuzzleSpace()

        #general parameters
        user = ExploratoryUser(problem_space)
        number_generation = 300 
        population_size = 200
        max_memory = 500 
        cross_over = 1 
        mutation = 0.05
        update_interval = 50


        #################################
        PARAMS = Parameters(seed=seed)
        #execution place
        now = datetime.now().strftime("%d-%m-%Y---%H-%M-%S")
        execFolder = f"{expFolder}/seed{PARAMS.seed}"
        os.makedirs(execFolder)
        PARAMS.execFolder = execFolder
        algo = YouAlgorithm(PARAMS, problem_space, number_generation, population_size, max_memory, cross_over, mutation, user, update_interval)
        for i in range(0,number_generation):
            algo.run_one_generation(False)
        # algo.run()
        # print("Average QD score: {}".format(algo.get_avg_qd_score()))
        # algo.save_measure_history(f"{algo.execFolder}/test_data")
        duration = time.time() - start_time
        return {"success": True, "seed": seed, "duration": duration}
    
    except Exception as e:
        duration = time.time() - start_time
        return {"success": False, "seed": seed, "duration": duration, "error": str(e)}


if __name__ == "__main__":
# 2. Definir a lista de tarefas (ex: diferentes sementes ou repetições)
    seeds = [124,4135,151256,631346,4515,131,51,6351,25,361]
    maxProcessors = 10 

    now = datetime.now().strftime("%d-%m-%Y---%H-%M-%S")
    expFolder = f"results/{now}"
    os.makedirs(expFolder)

    progressFilePath = f"{expFolder}/experiment_progress.txt"
    experimentStart = time.time()

    print(f"Started with {maxProcessors} processors...")

    allExecs = []
    for seed in seeds:
        allExecs.append((expFolder, seed))

    with Pool(processes=maxProcessors) as p:
        for result in p.imap_unordered(exec_wrapper, allExecs):
            status = "SUCCESS" if result["success"] else "FAILED"
            seed = result["seed"]
            duration = f"{result['duration']:.2f}s"
            elapsed = f"{time.time() - experimentStart:.2f}s"
            line = f"[{status}] Seed {seed} finished in {duration}; {elapsed} elapsed since start\n"
            print(line.strip())
            with open(progressFilePath, "a", encoding="utf-8") as f:
                f.write(line)