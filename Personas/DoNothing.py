import sys
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir)) 

import random 

from GeneticAlgorithmInterface import User
from ProblemSpaceInterface import ProblemSpace 

class DoNothing(User):
    def __init__(self, problem_space: ProblemSpace):
        self.problem_space = problem_space
    
    def update_constraints(self, cur_constraints, feasible):
        return cur_constraints , False