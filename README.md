# dharma-os-ai-agent
Dharma-OS: An autonomous AI Finance Agent built for the Scaler Bootcamp. It leverages FastAPI and Docker to manage financial tasks, achieving a 0.3 reward score in subscription management (e.g., Adobe cancellation). Deployed on Hugging Face Spaces with a robust, containerized backend architecture.
## Technical Overview

### 🏗 Architecture
The project follows a microservice-style architecture:
- **Backend**: FastAPI server running on Port 7860.
- **Environment**: Custom `DharmaEnv` that simulates financial states.
- **Containerization**: Fully Dockerized for seamless deployment.

### 📈 Performance
During evaluation, the agent successfully identified the high-cost Adobe subscription and initiated a cancellation sequence, resulting in a **Reward Score of 0.3**.

### 🛠 Installation & Local Setup
1. Clone the repo: `git clone https://github.com/Rojalin06/dharma-os-ai-agent`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `python -m server.app`
