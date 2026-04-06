from fastapi import FastAPI
import uvicorn
from env import DharmaEnv

app = FastAPI()
env = DharmaEnv()

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Dharma-OS is Running"}

@app.post("/reset")
def reset():
    try:
    obs, info = env.reset()
    return {"observation": obs, "info": info}
except Exception as e:
return {"error": str(e)}

@app.post("/step")
def step(action: dict):
    # Agent action handling
    obs, reward, done, info = env.step(action.get("action"))
    return {"observation": obs, "reward": reward, "done": done, "info": info}

def main():
    # Hugging Face ke liye port 7860 hona chahiye
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
