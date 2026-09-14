import os
import shutil
import time
from datetime import datetime
from multiprocessing import Pool


from parameters import Parameters
from ProblemSpaces.LodeRunner.LodeRunnerProblemSpace import LodRunnerProblemSpace
from ProblemSpaces.LogicPuzzles.LogicPuzzleSpace import LogicPuzzleSpace
# from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
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
    expFolder, seed = args
    start_time = time.time()

    try:
        problem_space = LogicPuzzleSpace()
        user = ExploratoryUser  (problem_space)

        PARAMS = Parameters(seed=seed)
        execFolder = f"{expFolder}/seed{PARAMS.seed}"
        os.makedirs(execFolder)
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
        # print("Average QD score: {}".format(algo.get_avg_qd_score()))
        algo.save_measure_history(f"{execFolder}/measureData.json")

        duration = time.time() - start_time
        return {"success": True, "seed": seed, "duration": duration, "avgQDscore":algo.get_avg_qd_score()}

    except Exception as e:
        import traceback
        duration = time.time() - start_time
        tb = traceback.format_exc()
        return {"success": False, "seed": seed, "duration": duration, "error": str(e), "traceback": tb, "avgQDscore":-1}


if __name__ == "__main__":
    # for i in range(2):
    #     expFolder = f"results/{datetime.now().strftime('%d-%m-%Y---%H-%M-%S')}"
    #     os.makedirs(expFolder)
    #     result = exec_wrapper((expFolder, 11))
    #     print(result)


    seeds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    maxProcessors = 14

    now = datetime.now().strftime("%d-%m-%Y---%H-%M-%S")
    expFolder = f"results/{now}"
    os.makedirs(expFolder)

    progressFilePath = f"{expFolder}/experiment_progress.txt"
    experimentStart = time.time()

    print(f"[{now}] Started with {maxProcessors} processors...")

    allExecs = [(expFolder, seed) for seed in seeds]
    QDscores = []
    seeds = []

    with Pool(processes=maxProcessors) as p:
        for result in p.imap_unordered(exec_wrapper, allExecs):
            QDscores.append(result["avgQDscore"])
            status = "SUCCESS" if result["success"] else "FAILED"
            seed = result["seed"]
            seeds.append(seed)
            duration = f"{result['duration']:.2f}s"
            elapsed = f"{time.time() - experimentStart:.2f}s"
            line = f"[{status}] Seed {seed} finished in {duration}; {elapsed} elapsed since start\n"
            print(line.strip())
            if not result["success"]:
                print(result["traceback"])
            with open(progressFilePath, "a", encoding="utf-8") as f:
                f.write(line)
                if not result["success"]:
                    f.write(result["traceback"] + "\n")
        with open(progressFilePath, "a", encoding="utf-8") as f:
            f.write(f"Avg QD score of all executions: {sum(QDscores)/len(QDscores)}")
            for i in range(len(seeds)):
                f.write(f"\nAvg QD score of seed {seeds[i]}: {QDscores[i]}")
