import pandas as pd
import matplotlib.pyplot as plt

def plot_fitness_metrics(csvFilepath: str) -> None:
    df = pd.read_csv(csvFilepath)
    df.columns = df.columns.str.strip()
    grouped = df.groupby('generation')['fitness'].agg(['max', 'mean']).reset_index()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Max fit
    ax1.plot(grouped['generation'], grouped['max'], color='tab:blue', marker='o', linewidth=2)
    ax1.set_title('Max fitness per Gen', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max fitness')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Avg fit
    ax2.plot(grouped['generation'], grouped['mean'], color='tab:orange', marker='s', linewidth=2)
    ax2.set_title('Average fitness per Gen', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Gen')
    ax2.set_ylabel('Average fitness')
    ax2.grid(True, linestyle='--', alpha=0.6)

    filepath = csvFilepath.split(".")[0]
    plt.savefig(f"{filepath}_fitness.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_fitness_metrics('results/10-09-2026---16-17-09.csv')