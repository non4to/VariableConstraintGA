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
from main import YouAlgorithm

class Algo(YouAlgorithm):
    """Subclass just for tests: adds saves in grid and bins in CSVs each gen"""

    def __init__(self, *args, exec_folder, **kwargs):
        super().__init__(*args, **kwargs)

        self.execFolder = exec_folder
        shutil.copy("parameters.py", self.execFolder)

        self.outputCSV = f"{self.execFolder}/data.csv"
        with open(self.outputCSV, mode='a', encoding='utf-8') as f:
            f.write("generation, x, y, fitness\n")

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
                    fit, solution = self.currentGrid[y][x]
                    f.write(f"{gen}, {x},{y}, {fit}\n")

    def save_bins_state(self, gen: int) -> None:
        output = f"\nGeneration: {gen},"
        for i, bin in enumerate(self.qualityBins):
            output += f"\n--Bin{i}:"
            for fit, _ in bin:
                output += f" {fit},"
        with open(self.binsCSV, mode='a', encoding='utf-8') as f:
            f.write(output)


def exec_wrapper(args: tuple) -> dict:
    expFolder, seed = args
    start_time = time.time()

    try:
        problem_space = LogicPuzzleSpace()
        user = ExploratoryUser(problem_space)
        number_generation = 300
        population_size = 200
        max_memory = 500
        cross_over = 1
        mutation = 0.05
        update_interval = 50

        PARAMS = Parameters(seed=seed)
        execFolder = f"{expFolder}/seed{PARAMS.seed}"
        os.makedirs(execFolder)
        PARAMS.execFolder = execFolder

        algo = Algo(
            PARAMS, problem_space, number_generation, population_size, max_memory,
            cross_over, mutation, user, update_interval,
            exec_folder=execFolder,
        )
        algo.run()
        print("Average QD score: {}".format(algo.get_avg_qd_score()))
        algo.save_measure_history(f"{execFolder}/measureData.json")

        duration = time.time() - start_time
        return {"success": True, "seed": seed, "duration": duration}

    except Exception as e:
        import traceback
        duration = time.time() - start_time
        tb = traceback.format_exc()
        return {"success": False, "seed": seed, "duration": duration, "error": str(e), "traceback": tb}


if __name__ == "__main__":
    expFolder = f"results/{datetime.now().strftime('%d-%m-%Y---%H-%M-%S')}"
    os.makedirs(expFolder)
    result = exec_wrapper((expFolder, 11))
    print(result)


    # seeds = [11]#,22,33,44,55,66,77,88,99,1010,1111,1212,1313,1414,1515]
    # maxProcessors = 10

    # now = datetime.now().strftime("%d-%m-%Y---%H-%M-%S")
    # expFolder = f"results/{now}"
    # os.makedirs(expFolder)

    # progressFilePath = f"{expFolder}/experiment_progress.txt"
    # experimentStart = time.time()

    # print(f"[{now}] Started with {maxProcessors} processors...")

    # allExecs = [(expFolder, seed) for seed in seeds]

    # with Pool(processes=maxProcessors) as p:
    #     for result in p.imap_unordered(exec_wrapper, allExecs):
    #         status = "SUCCESS" if result["success"] else "FAILED"
    #         seed = result["seed"]
    #         duration = f"{result['duration']:.2f}s"
    #         elapsed = f"{time.time() - experimentStart:.2f}s"
    #         line = f"[{status}] Seed {seed} finished in {duration}; {elapsed} elapsed since start\n"
    #         print(line.strip())
    #         if not result["success"]:
    #             print(result["traceback"])
    #         with open(progressFilePath, "a", encoding="utf-8") as f:
    #             f.write(line)
    #             if not result["success"]:
    #                 f.write(result["traceback"] + "\n")