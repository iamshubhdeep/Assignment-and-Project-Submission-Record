import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    rewards = []
    
    # Parse the Stable-Baselines3 Monitor log file
    with open("logs/training_monitor.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip metadata comments and the header row
            if row and not row[0].startswith("#"):
                if row[0] != "r": 
                    rewards.append(float(row[0]))

    # Calculate a moving average to clearly see the learning progression
    window_size = 20
    rolling_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode="valid")

    # Generate the performance plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward", alpha=0.3, color="royalblue")
    plt.plot(rolling_mean, label=f"Moving Average (w={window_size})", color="crimson", linewidth=2)
    
    plt.title("PPO Lunar Lander Training Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save the output visualization
    plt.savefig("graphs/learning_curve.png", dpi=300, bbox_inches="tight")
    print("Learning curve plot successfully saved to graphs/learning_curve.png")
    plt.show()

if __name__ == "__main__":
    main()