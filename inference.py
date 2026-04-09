import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv  # Ensure this matches your project structure
from models import Action

# 1. Mandatory Environment Variables (As per Tutorial 04)
API_BASE_URL = os.getenv("API_BASE_URL", "https://proxy.llm.scaler.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
# Scaler looks for HF_TOKEN to identify the participant
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

async def main():
    try:
        if not API_KEY:
            print("[ERROR] HF_TOKEN missing. Please add it to HF Secrets.")
            return

        # 2. Initialize OpenAI Client pointing to Proxy
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = DharmaEnv()

        # 3. Task Execution Loop
        tasks = ["task_1", "task_2", "task_3"] 
        
        for task_id in tasks:
            # [START] line is mandatory for validator
            print(f"[START] task={task_id} env=dharma_os model={MODEL_NAME}", flush=True)
            
            obs, info = env.reset(task_id=task_id)
            rewards = []
            
            # Step 1: LLM Logic
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": "You are a helpful OS assistant. Respond in JSON."},
                          {"role": "user", "content": f"Task: {task_id}. Observation: {obs}"}],
                response_format={ "type": "json_object" }
            )
            
            data = json.loads(response.choices[0].message.content)
            action = Action(
                category=data.get("category", "FINANCE"),
                command=data.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=data.get("target_id", "Unknown")
            )

            # Step 2: Environment Interaction
            obs, reward, done, info = await env.step(action)
            rewards.append(reward)
            
            # [STEP] line is mandatory immediately after env.step()
            # Format: reward must be 2 decimal places
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            # [END] line is mandatory after task completion
            avg_score = sum(rewards) / len(rewards)
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(f"[END] success={str(done).lower()} steps=1 score={avg_score:.3f} rewards={rewards_str}", flush=True)

    except Exception as e:
        # END log even on exception to avoid validator hang
        print(f"[END] success=false steps=0 score=0.000 rewards=0.00 error={str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
