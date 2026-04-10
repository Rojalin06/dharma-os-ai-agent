import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # MANDATORY: Directly using os.environ as requested by Scaler
        api_base = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        # Initialize Client pointing strictly to LiteLLM Proxy
        client = OpenAI(base_url=api_base, api_key=api_key)
        env = DharmaEnv()
        
        # REQUIRED LOGGING: [START], [STEP], [END]
        tasks = ["task_1", "task_2", "task_3"] 
        for task_id in tasks:
            print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)
            
            obs, info = env.reset(task_id=task_id)
            
            # API Call that MUST be logged by the proxy
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"Analyze: {obs}"}],
                response_format={ "type": "json_object" }
            )
            
            data = json.loads(response.choices[0].message.content)
            action = Action(
                category=data.get("category", "FINANCE"),
                command=data.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=data.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            
            # Formatting Reward to 2 decimal places is critical for the validator
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except KeyError as e:
        print(f"[CRITICAL ERROR] Missing Environment Variable: {e}", flush=True)
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
