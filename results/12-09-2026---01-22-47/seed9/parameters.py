import numpy as np

def individual_selection_v0(qualityBins:list ,solutions:list[tuple[float, object, int]], otherArgs:dict={}) -> tuple[float, object]:
    """Decides which individual from the list is selected to stay in the grid for the next generation.
    This function decides giving priority to solutions whose bins are not yet populated.
    It works with a given tolerance. The new-bin-solution has to be at least % of biggest solution of the list
    Solutions: list of tuples (fitness, solution object, bin)
    """
    tolerance = otherArgs["tolerance"]
    solutions = sorted(solutions, key=lambda solution: solution[0], reverse=True)
    for fit, solution, bin in solutions:
        if (len(qualityBins[bin]) == 0) and (fit >= solutions[0][0]*tolerance):
            return (fit, solution)
    return (solutions[0][0],solutions[0][1])

def select_parent2_random(randomGenerator:np.random.Generator, neighbors: list[tuple[int, int]], otherArgs: dict={}) -> tuple[int, int]:
    """Returns a random neighbor position from the list"""
    return randomGenerator.choice(neighbors)

PARAMETERS = {
    # "grid":[
    #     [1,1],
    #     [1,1]
    # ],
    "grid": [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ],
    "toroidal": False,
    "selectionFunc": individual_selection_v0,
    "parentSelec": select_parent2_random,
    "tolerance": 0.6,
}

class Parameters():
    def __init__(self, parameters: dict = PARAMETERS, seed: int = None):
        self.grid = parameters["grid"]
        self.toroidal = parameters["toroidal"]
        self.selectionFunc = parameters["selectionFunc"]
        self.parentSelectionFunc = parameters["parentSelec"]
        self.tolerance = parameters["tolerance"]
        self.expFolder = ""
        self.seed = seed
        if seed is None:
            self.random = np.random.default_rng()
        else:
            self.random = np.random.default_rng(seed)