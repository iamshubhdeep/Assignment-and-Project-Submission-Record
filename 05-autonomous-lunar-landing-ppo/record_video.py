import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

def main():
    # Ensure the videos directory exists
    os.makedirs("videos", exist_ok=True)

    # Initialize environment in rgb_array mode for video capture
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # Wrap the environment to trigger video recording for the episode
    env = RecordVideo(
        env, 
        video_folder="videos", 
        episode_trigger=lambda episode: True
    )
    
    # Load your trained model
    model = PPO.load("models/ppo_lunarlander")
    
    obs, info = env.reset()
    done = False
    
    print("--- Recording Episode Video ---")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    env.close()
    print("Video successfully saved in the videos/ folder.")

if __name__ == "__main__":
    main()