import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv  # Ensure this matches your project structure
from models import Action

async def main():
    try:
        # STRICT REQUIREMENT: Screenshot ke "HOW TO FIX" step 2 ke mutabiq exact code
        # os.environ[] use karne se validator ko 100% confirmation milti hai
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )
        
        # MODEL_NAME variable se uthayein
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        env = DharmaEnv()
        
        # REQUIRED LOGGING FORMAT FOR PHASE 2
        tasks = ["task_1", "task_2", "task_3"] 
        
        for task_id in tasks:
            # [START] line mandatory format
            print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)
            
            obs, info = env.reset(task_id=task_id)
            
            # LLM API Call through Proxy
            # Ensure the call is made using the proxy-configured client
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"Analyze state: {obs}. Task: {task_id}. Return JSON action."}],
                response_format={ "type": "json_object" }
            )
            
            content = json.loads(response.choices[0].message.content)
            action = Action(
                category=content.get("category", "FINANCE"),
                command=content.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=content.get("target_id", "Unknown")
            )

            # Execution
            obs, reward, done, info = await env.step(action)
            
            # [STEP] line mandatory format (reward format: 0.00)
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            # [END] line mandatory format
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except KeyError as e:
        # Agar variable nahi mila toh yahan print hoga
        print(f"[CRITICAL ERROR] Missing Environment Variable: {e}", flush=True)
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
