import os
import shutil
import time
import statistics
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool

from parameters import Parameters
from ProblemSpaces.LodeRunner.LodeRunnerProblemSpace import LodRunnerProblemSpace
from ProblemSpaces.LogicPuzzles.LogicPuzzleSpace import LogicPuzzleSpace
from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace

from Personas.Exploratory import ExploratoryUser
from Personas.DoNothing import DoNothing 
from Personas.Strict import StrictUser 
from Personas.Adaptive import AdaptiveUser
from Personas.TwoForwardOneBack import TwoForOneBackUser

from main import YouAlgorithm
from Algorithms.VCMapElites import VariableConstraintMapElites

class Algo(YouAlgorithm):
    """Subclass just for tests: adds saves in grid and bins in CSVs each gen"""

    def __init__(self, *args, exec_folder, **kwargs):
        super().__init__(*args, **kwargs)

        self.execFolder = exec_folder
        shutil.copy("parameters.py", self.execFolder)

        self.outputCSV = f"{self.execFolder}/data.csv"
        with open(self.outputCSV, mode='a', encoding='utf-8') as f:
            f.write("id, generation, x, y, isValid, fit, bin\n")

        self.binsCSV = f"{self.execFolder}/bins.csv"
        with open(self.binsCSV, mode='a', encoding='utf-8') as f:
            f.write("gen, bins\n")

        self.save_current_grid(0)

    def print_bins(self) -> None:
        for i, bin in enumerate(self.qualityBins):
            print(f"Bin {i} - {bin}")

    def print_grid(self) -> None:
        for y in self.currentGrid:
            for x in self.currentGrid[y]:
                print(f"({x},{y}) - {self.currentGrid[y][x]}")

    def run_one_generation(self, made_change):
        started = time.perf_counter()
        result = super().run_one_generation(made_change)
        finished = time.perf_counter()

        if (finished - started) > 30:
            print(f"Warning! Generation {self.currentGen} finished in {finished - started}")

        self.save_current_grid(self.currentGen)
        self.save_bins_state(self.currentGen)
        return result

    def save_current_grid(self, gen: int) -> None:
        with open(self.outputCSV, mode='a', encoding='utf-8') as f:
            for y in self.currentGrid:
                for x in self.currentGrid[y]:
                    solution = self.currentGrid[y][x]
                    f.write(f"{solution.id}, {gen}, {x},{y}, {solution.valid}, {solution.fit}, {solution.currentBin}\n")

    def save_bins_state(self, gen: int) -> None:
        output = f"\nGeneration: {gen},"
        for i, bin in enumerate(self.qualityBins):
            output += f"\n--Bin{i}:"
            for solution in bin:
                output += f" {solution.fit},"
        with open(self.binsCSV, mode='a', encoding='utf-8') as f:
            f.write(output)

def exec_wrapper(args: tuple) -> dict:
    expFolder, problemName, personaName, seed = args
    start_time = time.time()

    try:
        problem_space = PROBLEMS[problemName]()
        user = PERSONAS[personaName](problem_space)

        PARAMS = Parameters(seed=seed)
        execFolder = f"{expFolder}/{problemName}/{personaName}/seed{PARAMS.seed}"
        os.makedirs(execFolder, exist_ok=True)
        
        PARAMS.execFolder = execFolder
        number_generation = PARAMS.number_generation
        population_size = PARAMS.population_size
        max_memory = PARAMS.max_memory
        cross_over = PARAMS.cross_over
        mutation = PARAMS.mutation
        update_interval = PARAMS.update_interval

        algo = Algo(
            PARAMS, problem_space, number_generation, population_size, max_memory,
            cross_over, mutation, user, update_interval,
            exec_folder=execFolder,
        )
        
        algo.run()
        algo.save_measure_history(f"{execFolder}/measureData.json")

        duration = time.time() - start_time
        return {
            "success": True, 
            "problemName": problemName,
            "personaName": personaName,
            "seed": seed, 
            "duration": duration, 
            "avgQDscore": algo.get_avg_qd_score()
        }

    except Exception as e:
        import traceback
        duration = time.time() - start_time
        tb = traceback.format_exc()
        return {
            "success": False, 
            "problemName": problemName,
            "personaName": personaName,
            "seed": seed, 
            "duration": duration, 
            "error": str(e), 
            "traceback": tb, 
            "avgQDscore": -1
        }

def run_batch(execList:list, processors:int):
        if not execList: return
        print(f"\n--- Rodando {len(execList)} tarefas em ({processors} processador(es)) ---")
        with Pool(processes=processors) as p:
            for result in p.imap_unordered(exec_wrapper, execList):
                status = "SUCCESS" if result["success"] else "FAILED"
                prob = result["problemName"]
                pers = result["personaName"]
                seed = result["seed"]
                duration = f"{result['duration']:.2f}s"
                elapsed = f"{time.time() - experimentStart:.2f}s"
                
                line = f"[{status}] {prob} | {pers} | Seed {seed} em {duration}; Total: {elapsed}\n"
                print(line.strip())
                resultSummary.append(result)
                with open(progressFilePath, "a", encoding="utf-8") as f:
                    f.write(line)
                    if not result["success"]:
                        f.write(result["traceback"] + "\n")

if __name__ == "__main__":
    PROBLEMS = {
    "LodeRunner": LodRunnerProblemSpace,
    "LogicPuzzle": LogicPuzzleSpace,
    "TTP": TTPProblemSpace
    }

    PERSONAS = {
        "Exploratory": ExploratoryUser,
        "DoNothing": DoNothing,
        "Strict": StrictUser,
        "Adaptive": AdaptiveUser,
        "TwoForwardOneBack": TwoForOneBackUser
    }

    SEEDS = [1,2,3,4,5]
    maxProcessors = 10  
    now = datetime.now().strftime("%d-%m-%Y---%H-%M-%S")
    expFolder = f"results/{now}"
    os.makedirs(expFolder, exist_ok=True)

    progressFilePath = f"{expFolder}/experiment_progress.txt"
    experimentStart = time.time()

    allExecs = []
    for problemName in PROBLEMS.keys():
        for personaName in PERSONAS.keys():
            for seed in SEEDS:
                allExecs.append((expFolder, problemName, personaName, seed))

    ttpExecs = [e for e in allExecs if e[1] == "TTP"]
    otherExecs = [e for e in allExecs if e[1] != "TTP"]

    print(f"[{now}] Iniciando {len(allExecs)} experimentos ({len(PROBLEMS)} problemas x {len(PERSONAS)} personas x {len(SEEDS)} seeds)...")
    print(f"Usando {maxProcessors} processos em paralelo.")
    resultSummary = []

    #parellelize everything that is not TTP
    run_batch(otherExecs, maxProcessors)
    run_batch(ttpExecs, 1) #TTP is not paralelized

    # Results stuff
    scoresGrouped = defaultdict(list)
    for res in resultSummary:
        if res["success"]:
            scoresGrouped[(res["problemName"], res["personaName"])].append(res["avgQDscore"])

    header1 = "\n=== SUMMARY PER SEED ===\n"
    print(header1.strip())
    with open(progressFilePath, "a", encoding="utf-8") as f:
        f.write(header1)
        for res in resultSummary:
            line = f"{res['problemName']} | {res['personaName']} | Seed {res['seed']} -> QD Score: {res['avgQDscore']:.4f}\n"
            f.write(line)

    header2 = "\n=== FULL RUN SUMMARY ===\n"
    print(header2.strip())

    with open(progressFilePath, "a", encoding="utf-8") as f:
        f.write(header2)
        for (prob, pers), scores in scoresGrouped.items():
            meanValue = statistics.mean(scores)
            stdValue = statistics.stdev(scores) if len(scores) > 1 else 0.0
            
            line = f"{prob} | {pers} -> Avg: {meanValue:.4f} | Std.Dev: {stdValue:.4f} (N={len(scores)})\n"
            print(line.strip())
            f.write(line)