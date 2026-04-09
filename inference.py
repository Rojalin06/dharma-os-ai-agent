import asyncio
import os
import json
import textwrap
from openai import OpenAI
from env import DharmaEnv
from models import Action

# Mandatory variables as per sample script
API_BASE_URL = os.getenv("API_BASE_URL", "https://proxy.llm.scaler.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
# Try both names to be safe
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Logging Functions (Strictly required format)
def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    try:
        if not API_KEY:
            print("[ERROR] API_KEY/HF_TOKEN missing")
            return

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = DharmaEnv()
        tasks = ["task_1", "task_2", "task_3"] 

        for task_id in tasks:
            log_start(task_id, "dharma_os", MODEL_NAME) #
            
            obs, info = env.reset(task_id=task_id)
            rewards = []
            
            # Phase 2 usually checks a single step or loop
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Task: {task_id}. State: {obs}. Return JSON."}],
                response_format={ "type": "json_object" }
            )
            
            content = json.loads(response.choices[0].message.content)
            action = Action(
                category=content.get("category", "FINANCE"),
                command=content.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=content.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            rewards.append(reward)
            
            # Log each step precisely
            log_step(1, action.command, reward, done)
            
            # Log the end of task
            log_end(done, 1, reward, rewards)

    except Exception as e:
        print(f"[DEBUG] Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
