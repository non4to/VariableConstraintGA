import sys
import os 
from pathlib import Path
import math 
from copy import deepcopy
import numpy as np 
 


sys.path.append(str(Path.cwd().parent.parent)) 
sys.path.append(str(Path.cwd().parent)) 
sys.path.append(str(Path.cwd())) 
from ProblemSpaceInterface import ProblemSpace, Constraint

from ProblemSpaces.TravelingThief.GeneticOperators import random_individual, mutate, cross_over 
from ProblemSpaces.TravelingThief.Utils import read_file, fitness 
from ProblemSpaces.TravelingThief.Constraints import WeightConstraint, rand_constraint_ind, random_constraint, is_contradictory
BASE_DIR = Path(__file__).resolve().parent

rawProblems = ["n50_bounded_strongly.ttp", "n50_uncorr_similar.ttp", "n50_uncorr.ttp", "n150_strongly_bounded.ttp", "n150_uncorr_similar.ttp", "n150_uncorr.ttp"]

problems = [str(BASE_DIR / problem) for problem in rawProblems]
nextProblemFile = BASE_DIR / "nextProblem.txt"
# problems = ["../ProblemSpaces/TravelingThief/" + problem for problem in problems]

# nextProblemFile = "../ProblemSpaces/TravelingThief/nextProblem.txt"

class TTPIndivual():
    def __init__(self, params, nodes, items, solution) -> None:
        self.solution = solution 
        self.params = params 
        self.nodes = nodes 
        self.items = items 
        self.fit = None 
    def get_fit(self):
        if self.fit is None:
            self.fit = fitness(self.params, self.nodes, self.items, self.solution)
        
        return self.fit 


class TTPProblemSpace(ProblemSpace): 
    """
    To add a problem space to this benchmark, 
    create a class that inherits this class as a 
    parent 

    You will need to over-write all methods 
    in this class 
    """
    def __init__(self): 
        print(open(nextProblemFile, "r").read().strip())
        problem_int = int(open(nextProblemFile, "r").read().strip()) 
        problem = problems[problem_int]
        next_problem = (problem_int + 1 ) % len(problems)
        file = open(nextProblemFile, "w")
        file.write(str(next_problem))
        file.close()
        self.params, self.nodes, self.items = read_file(problem)

    def generate_random_individual(self):
        '''
        Return a random in individual in the problem space 
        for initialization 
        '''
        return  TTPIndivual(self.params, self.nodes, self.items, random_individual(self.params["cities"], self.params["num_items"]))

    def mutate(self, individual, mutation_rate):
        '''
        Take in an individual and return a mutation

        Amount of mutation can vary between 0 and 1 from mutation rate 
        e.g. each bit has a X% chance of flipping 

        Should NOT modify starting individual 
        '''
   
        return TTPIndivual(self.params, self.nodes, self.items,  mutate(individual.solution, mutation_rate))

    def cross_over(self, ind1, ind2):
        """
        Take in two individuals and return a random combination of the two

        Should NOT modify the starting individuals  

        """
        sol1, sol2 =  cross_over(ind1.solution, ind2.solution) 
        return TTPIndivual(self.params, self.nodes, self.items,  sol1), TTPIndivual(self.params, self.nodes, self.items,  sol2)
    
    def fitness(self, ind): 
        """
        returns how close the percentage of items matches 
        with the percentages of the original games 
        """
        return ind.get_fit()["fitness"]
    
    def get_num_bins(self):
        """
        Return the number of bins of the diversity measure 
        """
        return 10 
    
    def place_in_bin(self, ind):
        """
        Return an index between 0 and 9
        on how much profit vs distance is optimized 

        the left most bins will have large profit - but also large distance 
        the right most bins will have low distance time but low profit 
        """
        fit = ind.get_fit()

        # find how close profit is to optimal 
        profit_ratio = fit["total_profit"] / self.params["max_profit"]
        profit_ratio = min(1, profit_ratio)
        

        # find how close time is to optimal 
        time_ratio = (self.params["max_distance"] - fit["total_distance"]) / (self.params["max_distance"] - self.params["min_distance"])
        time_ratio = min(time_ratio, 1)
        time_ratio = max(time_ratio, 0)

        # trade off value 
        trade_off = profit_ratio / (profit_ratio + time_ratio)

        bi = math.floor(trade_off * 10) -1
        
        return bi 

    def get_constant_constraints(self):
        """
        Return a list of constraints that 
        all individuals must obey

        These constraints should be ATOMIC rather then general 
        """
        ratios = list(np.arange(1, 4, 0.1 ))

        constraints = [WeightConstraint(self.params, self.nodes, self.items, ratio) for ratio in ratios]

        return constraints
    
    def get_initial_variable_constraints(self):
        """
        Return a list of initial variable constraints 
        """
        return [] 
    

    def is_contradictory(self, constraints):
        """
        Given a list of constraints 
        return true if they are inherently 
        contradictory 

        This does NOT need to ensure 
        that an individual could exist with 
        all listed constraints 
        """
        return is_contradictory(constraints, self.items, self.params["knapsack_capacity"])  
    
    def get_rand_constraint(self):
        """
        Return a random variable constraint from the 
        problem space 
        """
        return random_constraint(self.params["num_items"])
    
    def get_ind_constraint(self, ind):
        """
        Given an individual, return a random 
        constraint that is is obeys 
        """
        return rand_constraint_ind(ind)
    

    