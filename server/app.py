from fastapi import FastAPI
import uvicorn
from env import DharmaEnv

app = FastAPI()
env = DharmaEnv()

# Yeh endpoint Hugging Face ke health check ke liye zaroori hai
@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Dharma-OS is Running"}

@app.post("/step")
def step(action: dict):
    # Action handling logic
    obs, reward, done, info = env.step(action.get("action"))
    return {"observation": obs, "reward": reward, "done": done, "info": info}

def main():
    # Host aur Port bilkul yahi hone chahiye
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
