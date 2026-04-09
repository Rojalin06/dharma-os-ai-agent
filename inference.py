import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # MANDATORY: Screenshot ke "HOW TO FIX" step 2 ke mutabiq exact initialization
        # Hum direct os.environ use kar rahe hain jaisa Scaler ne manga hai
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )
        
        # MODEL_NAME ko variable se uthayein
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        env = DharmaEnv()
        
        # MANDATORY LOGGING FORMAT
        tasks = ["task_1", "task_2", "task_3"] 
        
        for task_id in tasks:
            # [START] line mandatory format
            print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)
            
            obs, info = env.reset(task_id=task_id)
            
            # LLM API Call through Proxy
            response = client.chat.completions.create(
                model=model_name,
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
            
            # [STEP] line mandatory format immediately after step
            # Reward must be 0.0-1.0 range
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            # [END] line mandatory format
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except KeyError as e:
        print(f"[ERROR] Missing Variable: {e}. Check HuggingFace Secrets.")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    asyncio.run(main())
