import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

def main():
    # Make sure output dirs exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Env setup
    env = gym.make("LunarLander-v3")
    env = Monitor(env, filename="logs/training_monitor.csv")

    # Baseline hyperparameters from documentation
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=2048,
        n_epochs=10,
        clip_range=0.2,
        verbose=1
    )

    print("--- Starting PPO Training Loop ---")
    model.learn(total_timesteps=500000)
    
    # Save policy weights
    model.save("models/ppo_lunarlander")
    print("Training finished. Model saved.")

if __name__ == "__main__":
    main()