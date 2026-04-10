import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # MANDATORY: Scaler environment variables
        # Inhe directly client mein pass karna zaroori hai
        api_base = os.environ.get("API_BASE_URL")
        # Scaler portal says use API_KEY
        api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        if not api_key or not api_base:
            print("[ERROR] Environment variables are missing!", flush=True)
            return

        # Initialize OpenAI Client pointing strictly to LiteLLM Proxy
        client = OpenAI(
            base_url=api_base, 
            api_key=api_key
        )

        env = DharmaEnv()
        
        # REQUIRED LOGGING FORMAT: [START], [STEP], [END]
        tasks = ["task_1", "task_2", "task_3"] 
        
        for task_id in tasks:
            print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)
            
            obs, info = env.reset(task_id=task_id)
            
            # API Call - Iska track LiteLLM rakhta hai
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
            
            # [STEP] log is mandatory after each step
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            # [END] log is mandatory after each task
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
