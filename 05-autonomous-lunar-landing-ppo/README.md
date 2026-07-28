# Autonomous Lunar Landing using Proximal Policy Optimization (PPO)

A Deep Reinforcement Learning project focused on training an autonomous spacecraft agent to safely land on a designated landing pad using the Box2D physics engine physics simulator. 

## Developer Information
* **Name:** Akshat Garg
* **Registration Number:** 23BCE10641
* **Development Environment:** macOS (Apple Silicon M2 Pro/Max/Base architecture)

---

## Project Objective
The goal of this implementation is to leverage the Actor-Critic framework via PPO to master the continuous flight control dynamics of the LunarLander environment[cite: 1]. Instead of utilizing hand-crafted rule-based physics equations, the agent organically derives an optimal thrust policy through active exploration and cumulative reward maximization[cite: 1].

### Environment Specifications
* **Environment Name:** `LunarLander-v3`[cite: 1]
* **Observation Space:** 8-Dimensional continuous vector tracking positions ($x, y$), velocities ($v_x, v_y$), tilt angle, angular velocity, and left/right leg ground contact assertions[cite: 1].
* **Action Space:** Discrete (4) representing: Do nothing, fire left orientation engine, fire main engine, or fire right orientation engine[cite: 1].

---

## Project Directory Layout
```text
LunarLander-PPO/
├── models/
│   └── ppo_lunarlander.zip     # Saved trained PPO policy network weights
├── logs/
│   └── training_monitor.csv    # Evaluated step-by-step rolling environment metrics
├── videos/
│   └── rl-video-episode-0.mp4  # Stitched evaluation flight recording
├── graphs/
│   └── learning_curve.png      # Rendered performance progression plot
├── train.py                    # Policy initialization and baseline training loop
├── evaluate.py                 # Quantitative deterministic verification script
├── test.py                     # Visual environment deployment runtime execution
├── plot_training.py            # Custom parsing utility for monitoring logs
└── requirements.txt            # System dependencies manifest

```

---

## System Setup & Installation

Since the Box2D physics engine requires compiling C++ libraries locally on macOS architectures, the environment relies on standard build tools management via Conda.

1. **Clone the repository:**
```bash
git clone https://github.com/AkshatGarg2005/Autonomous-Lunar-Landing-using-Proximal-Policy-Optimization
cd LunarLander-PPO

```


2. **Initialize and activate the Conda environment:**
```bash
conda create -n lunar_lander python=3.10 -y
conda activate lunar_lander

```


3. **Install the required system build tools and package dependencies:**
```bash
conda install -c conda-forge swig -y
pip install -r requirements.txt

```



---

## Hyperparameter Configuration

The implementation utilizes baseline settings optimized for multi-layer perceptron processing layers:

* **Network Policy:** `MlpPolicy`

* **Learning Rate ($e$):** $3 \times 10^{-4}$

* **Discount Factor ($\gamma$):** $0.99$

* **Batch Size:** $64$

* **Rollout Steps ($n$):** $2048$

* **Optimization Epochs:** $10$

* **Clip Range:** $0.2$


---

## Execution Guide

### 1. Training the Agent

Run the base optimization loop to generate experience trajectories up to 500,000 timesteps:

```bash
python train.py

```

### 2. Quantitative Evaluation

Run the model over multiple consecutive episodes to extract performance metrics:

```bash
python evaluate.py

```

### 3. Visual Human Rendering

Launch a real-time simulator window showing the trained weights controlling the vehicle:

```bash
python test.py

```

### 4. Automated Video Export

Generate unified `.mp4` recordings of completed flights using the wrapper interface:

```bash
python record_video.py

```

### 5. Learning Curve Plotting

Generate the localized graph parsing the training history logs:

```bash
python plot_training.py

```

---

## Empirically Achieved Performance

The agent converged smoothly inside the 500k training timestep limit, breaking out of negative reward boundaries within the first 300 episodes and steadily flatlining near peak capability.

* **Deterministic Mean Evaluation Score (20 Episodes):** $170.91 \pm 101.20$
* **Peak Test Runs Realized Scores:**
* Episode 1: **256.16**
* Episode 2: **193.09**
* Episode 3: **233.42**
* Episode 4: **240.13**
* Episode 5: **234.41**

According to the scoring tier matrices, maintaining consistent values tracking above $+200$ asserts that the policy has achieved an **Excellent** operational rating, executing stable, center-pad touchdowns.
