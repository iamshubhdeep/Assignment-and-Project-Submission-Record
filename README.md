# ML / AI / RL Project Portfolio

A consolidated collection of machine learning, deep learning, and reinforcement learning projects — classic ML classifiers, computer vision, NLP/RAG, and RL agents.

## Projects

| # | Project | Type | Notes |
|---|---------|------|-------|
| 01 | [Cipher-10](./01-cipher-10) | Deep Learning (Notebook) | Image classification on CIFAR-10 |
| 02 | [Face Recognition](./02-face-recognition) | Computer Vision (Notebook) | Face detection/recognition pipeline |
| 03 | [Cancer Classification](./03-cancer-classification) | ML Classification (Notebook) | Diagnostic classifier |
| 04 | [Adult Census ML](./04-adult-census-ml) | ML Classification (Script) | Income prediction on the Adult Census dataset |
| 05 | [Autonomous Lunar Landing (PPO)](./05-autonomous-lunar-landing-ppo) | Reinforcement Learning | PPO agent solving LunarLander |
| 06 | [Movie Recommendation System](./06-movie-recommendation-system) | ML / Web App | [Live demo](https://movie-recommendation-system-oo9b.onrender.com) |
| 07 | [RAG Chatbot](./07-rag-chatbot) | LLM / NLP | Retrieval-augmented chatbot — [Live demo](https://rag-chatbot-blei.onrender.com) |
| 08 | [Iris Flower Classifier (API + Web UI)](./08-iris-flower-classifier) | ML / Web App | [Live demo](https://iris-flower-classifier-api-web-ui.onrender.com) |
| 09 | [Cart-Pole RL Trainer](./09-cartpole-rl-trainer) | Reinforcement Learning (Interactive) | Browser-based Q-learning visualizer — open the `.html` file directly |

## Structure

Each project lives in its own folder with its original code/notebooks intact. Projects 05–08 were pulled in from their own standalone repositories (see individual folder READMEs for their original setup/run instructions). Projects 01–04 and 09 are notebooks/scripts/artifacts added directly.

## Running things

- **Notebooks (01–03)**: open in Jupyter / Colab and run top to bottom.
- **04-adult-census-ml**: `python adult_census_ml_assignment.py` (see `images/` for saved output plots/tables).
- **05–08**: each has its own `requirements.txt` — `cd` into the folder, `pip install -r requirements.txt`, then follow that folder's README.
- **09-cartpole-rl-trainer**: open `cartpole_rl_trainer.html` directly in a browser, no install needed.
