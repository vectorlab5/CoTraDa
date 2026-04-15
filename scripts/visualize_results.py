import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_metrics(json_path: str, output_dir: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    episodes = [d['episode'] for d in data]
    rewards = [d['reward'] for d in data]
    utilities = [d['data_utility'] for d in data]
    aois = [d['weighted_aoi'] for d in data]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(episodes, rewards, label='Reward', color='blue')
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].legend()
    
    axes[1].plot(episodes, utilities, label='Utility', color='green')
    axes[1].set_title('Data Utility')
    axes[1].set_xlabel('Episode')
    axes[1].legend()
    
    axes[2].plot(episodes, aois, label='Weighted AoI', color='red')
    axes[2].set_title('Weighted AoI')
    axes[2].set_xlabel('Episode')
    axes[2].legend()
    
    plt.tight_layout()
    out_path = Path(output_dir) / "training_curves.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_results.py <json_path> <output_dir>")
    else:
        plot_metrics(sys.argv[1], sys.argv[2])
