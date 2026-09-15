import numpy as np

def individual_selection_v0(qualityBins:list ,solutions:list["Solution"], otherArgs:dict={}) -> "Solution":
    """Decides which individual from the list is selected to stay in the grid for the next generation.
    This function decides giving priority to solutions whose bins are not yet populated.
    It works with a given tolerance. The new-bin-solution has to be at least % of biggest solution of the list
    Solutions: list of tuples (fitness, solution object, bin)
    """
    tolerance = otherArgs["tolerance"]
    solutions = sorted(solutions, key=lambda solution: solution.fit, reverse=True)
    for solution in solutions:
        if (len(qualityBins[solution.currentBin]) < 1) and (solution.fit >= solutions[0].fit*tolerance):
            #return solution if there is no solution in the solution's bin and solution fitness is a minimum of its pears. else, just returns the best
            return solution
    return solutions[0]

def select_parent2_random(randomGenerator:np.random.Generator, neighbors: list[tuple[int, int]], otherArgs: dict={}) -> tuple[int, int]:
    """Returns a random neighbor position from the list"""
    return randomGenerator.choice(neighbors)

PARAMETERS = {
    # "grid":[
    #     [1,1],
    #     [1,1]
    # ],
    "grid": [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ],
    "mutationGrid": [ 
    [0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5] ,
    [0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1] ,
    [0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1] ,
    [0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5] ,
    [0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1] ,
    [0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1] ,
    [0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5] ,
    [0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1] ,
    [0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1] ,
    [0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5] 
    ],
    "toroidal": True,
    "selectionFunc": individual_selection_v0,
    "parentSelec": select_parent2_random,
    "tolerance": 0.5,
    "number_generation":300,
    "max_memory":500,
    "cross_over":0.5,
    "mutation":0.1,
    "update_interval":50,
    "useMutationGrid":True,
}
PARAMETERS["population_size"] = sum(len(line) for line in PARAMETERS["grid"])

class Parameters():
    def __init__(self, parameters: dict = PARAMETERS, seed: int = None):
        self.grid = parameters["grid"]
        self.mutationGrid = parameters["mutationGrid"]
        self.useMutationGrid = parameters["useMutationGrid"]
        self.toroidal = parameters["toroidal"]
        self.selectionFunc = parameters["selectionFunc"]
        self.parentSelectionFunc = parameters["parentSelec"]
        self.tolerance = parameters["tolerance"]
        self.number_generation = parameters["number_generation"]
        self.population_size = parameters["population_size"]
        if self.population_size > 200: raise ValueError("Population size must be equal or under 200.")
        self.max_memory = parameters["max_memory"]
        if self.max_memory > 500: raise ValueError("Max memory must be 500 maximum.")
        self.cross_over = parameters["cross_over"]
        self.mutation = parameters["mutation"]
        self.update_interval = parameters["update_interval"]
        self.expFolder = ""
        self.seed = seed
        if seed is None:
            self.random = np.random.default_rng()
        else:
            self.random = np.random.default_rng(seed)