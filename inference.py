import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

# 1. Variables exactly as requested in "How to Fix"
# Direct os.environ use karein taaki validator ko clear signal mile
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

async def main():
    try:
        # 2. Initialize OpenAI client pointing to the proxy
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

        env = DharmaEnv()
        tasks = ["task_1", "task_2", "task_3"] 
        
        for task_id in tasks:
            # MANDATORY LOGGING FORMAT
            print(f"[START] task={task_id} env=dharma_os model={MODEL_NAME}", flush=True)
            
            obs, info = env.reset(task_id=task_id)
            
            # API Call that MUST hit the LiteLLM proxy
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Analyze: {obs}"}],
                response_format={ "type": "json_object" }
            )
            
            content = json.loads(response.choices[0].message.content)
            action = Action(
                category=content.get("category", "FINANCE"),
                command=content.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=content.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            
            # [STEP] line mandatory format (reward as 2 decimal places)
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            # [END] line mandatory format
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
