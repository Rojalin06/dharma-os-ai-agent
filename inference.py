import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # 1. INITIALIZATION: Exactly as per "HOW TO FIX" Step 2
        # Using direct environ access to satisfy the validator's check
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )
        
        # Model name from environment
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        env = DharmaEnv()
        
        # REQUIRED LOGGING FORMAT: [START], [STEP], [END]
        tasks = ["task_1", "task_2", "task_3"] 
        
        for task_id in tasks:
            # Mandatory [START] line
            print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)
            
            obs, info = env.reset(task_id=task_id)
            
            # API Call - Must use the 'client' configured with proxy
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"Analyze state: {obs}. Return JSON."}],
                response_format={ "type": "json_object" }
            )
            
            data = json.loads(response.choices[0].message.content)
            action = Action(
                category=data.get("category", "FINANCE"),
                command=data.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=data.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            
            # Mandatory [STEP] line (format: 2 decimal places)
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            # Mandatory [END] line
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except KeyError as e:
        print(f"[CRITICAL ERROR] Missing variable: {e}. Check HF Secrets.", flush=True)
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
