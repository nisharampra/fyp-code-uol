# fyp-code-uol
# Financial Advisor Bot: A Simulation-Based Educational Platform for Algorithmic Trading Literacy

## Overview
The Financial Advisor Bot is a full-stack educational simulator that combines LSTM forecasting, Deep Q-Network (DQN) reinforcement learning, and explainable AI (LLaMA3 via Ollama).  
It allows users to interact with historical financial data, receive AI trading recommendations, override them, and get natural-language explanations. The goal is not just profit, but teaching responsible financial decision-making in a safe environment.

## Features
- LSTM Forecasting: Predicts short-term stock movements.  
- DQN Trading Agent: Learns trading strategies via reinforcement learning.  
- Explainability: Natural language explanations from a local LLaMA3 model (via Ollama).  
- Interactive Dashboard: Users can log in, select tickers, place trades, and see AI feedback.  
- Simulation Mode: Safe practice environment with detailed post-session feedback.  
- Evaluation Tools: Trade logs, quizzes, and feedback surveys to measure learning.

## Tech Stack
- Backend: Flask (Python)  
- Frontend: Node.js (Express + EJS templates)  
- Database: SQLite  
- Machine Learning: PyTorch, pandas, ta, stable-baselines3  
- Authentication: bcrypt for secure login  
- Explainability: LLaMA3 via Ollama (local LLM integration)  

## Project Structure
fyp-final-prototype/
│── api.py                  # Flask backend API
│── server.js               # Node.js frontend (Express + EJS)
│── models/                 # ML models (LSTM, DQN)
│── static/                 # CSS, JS, client assets
│── templates/              # EJS templates
│── data/                   # Historical stock datasets
│── tests/                  # Unit, model, and integration tests
│── .gitattributes          # Git LFS tracking for large files
│── requirements.txt        # Python dependencies
│── package.json            # Node.js dependencies

## Installation & Setup

### 1. Clone the Repository
git clone git@github.com:nisharampra/fyp-code-uol.git
cd fyp-code-uol

### 2. Backend (Flask API)
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Flask API
python api.py

### 3. Frontend (Node.js)
# Install dependencies
npm install

# Run frontend server
node server.js

## Testing
- Python Tests (LSTM, DQN, Flask API):
  pytest -q

- Node.js Tests (routes, auth, UI):
  npm test

## Evaluation
The system has been rigorously evaluated using:
- Unit Testing: Verification of API, models, and UI components.  
- Model Testing: Forecast accuracy, DQN trading policy, robustness checks.  
- Backtesting: Benchmarked against MA crossover, RSI, MACD strategies.  
- User Testing: Quizzes, surveys, and comprehension feedback.  
- Load Testing: Stability under multiple concurrent sessions.  
- Risk Testing: Stress tests via Monte Carlo simulations.

## Report
For a full academic write-up (Introduction, Literature Review, Design, Implementation, Evaluation, Conclusion), see the attached Final Year Project Report.
