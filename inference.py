import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

# 1. Mandatory Variables - Directly as per Scaler Checklist
# Make sure your HF Secret name is 'API_KEY'
API_BASE_URL = os.getenv("API_BASE_URL", "https://proxy.llm.scaler.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
API_KEY = os.getenv("API_KEY") # Prioritize this as per 'How to Fix' instructions

async def main():
    try:
        if not API_KEY or not API_BASE_URL:
            print("[ERROR] Mandatory environment variables missing!")
            return

        # 2. Client Initialization exactly as per instruction Step 2
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = DharmaEnv()

        tasks = ["task_1", "task_2", "task_3"] 
        for task_id in tasks:
            # REQUIRED: [START] line format
            print(f"[START] task={task_id} env=dharma_os model={MODEL_NAME}", flush=True)
            
            obs, info = env.reset(task_id=task_id)
            rewards = []
            
            # API Call that MUST go through proxy
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Task: {task_id}. Observation: {obs}"}],
                response_format={ "type": "json_object" }
            )
            
            data = json.loads(response.choices[0].message.content)
            action = Action(
                category=data.get("category", "FINANCE"),
                command=data.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=data.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            rewards.append(reward)
            
            # REQUIRED: [STEP] line format - EXACT FIELD NAMES
            # reward must be formatted to 2 decimal places
            # done must be lowercase true/false
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            # REQUIRED: [END] line format
            avg_reward = sum(rewards) / len(rewards)
            print(f"[END] success={str(done).lower()} steps=1 score={avg_reward:.3f} rewards={reward:.2f}", flush=True)

    except Exception as e:
        # Avoid crashing so validator can see the log
        print(f"[END] success=false steps=0 score=0.000 rewards=0.00 error={str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
