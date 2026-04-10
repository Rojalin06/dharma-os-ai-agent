import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # 1. STRICT INITIALIZATION [As per image_422166.jpg instructions]
        # Direct environ access ensures the proxy variables are picked up
        api_base = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        # Creating the OpenAI Client
        client = OpenAI(base_url=api_base, api_key=api_key)
        
        env = DharmaEnv()
        
        # 2. REQUIRED LOGGING FORMAT [Mandatory for Scaler Validator]
        tasks = ["task_1", "task_2", "task_3"] 
        for task_id in tasks:
            print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)
            
            obs, info = env.reset(task_id=task_id)
            
            # 3. LLM Call via LiteLLM Proxy
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"Task: {task_id}. State: {obs}."}],
                response_format={ "type": "json_object" }
            )
            
            content = json.loads(response.choices[0].message.content)
            action = Action(
                category=content.get("category", "FINANCE"),
                command=content.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=content.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            
            # MANDATORY: Reward with 2 decimal places and lowercase 'done'
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except KeyError as e:
        print(f"[CRITICAL ERROR] Variable {e} missing on Hugging Face Secrets!", flush=True)
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
