import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

# STRICT MANDATORY CONFIGURATION
# Scaler injects these names. Do NOT use default strings here.
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
# Use exact name check for the key
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

async def main():
    try:
        # Step 1: Client initialization strictly via proxy
        if not API_KEY or not API_BASE_URL:
            print(f"[ERROR] Environment variables missing: URL={bool(API_BASE_URL)}, KEY={bool(API_KEY)}")
            return

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

        env = DharmaEnv()
        
        # Mandatory Start Log
        print(f"[START] task=task_1 env=dharma_os model={MODEL_NAME}", flush=True)

        # Step 2: Running the tasks
        tasks = ["task_1", "task_2", "task_3"]
        for task_id in tasks:
            obs, info = env.reset(task_id=task_id)
            
            # This is the call that MUST go through the proxy
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Task: {task_id}. State: {obs}. Respond in JSON."}],
                response_format={ "type": "json_object" }
            )
            
            data = json.loads(response.choices[0].message.content)
            action = Action(
                category=data.get("category", "FINANCE"),
                command=data.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=data.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            
            # Step 3: Strict Output Format
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except Exception as e:
        # Printing error helps debug if something crashes
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
