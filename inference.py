import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

# Documentation ke mutabiq mandatory variables
# Isme '/v1' hona zaroori hai agar LiteLLM use ho raha hai
API_BASE_URL = os.getenv("API_BASE_URL", "https://proxy.llm.scaler.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
# Scaler injects either API_KEY or HF_TOKEN
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

async def main():
    try:
        if not API_KEY:
            print("[ERROR] HF_TOKEN/API_KEY not found in environment!")
            return

        # OpenAI Client configuration strictly via Scaler Proxy
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

        env = DharmaEnv()
        
        # Mandatory Start Log
        print(f"[START] task=task_1 env=dharma_os model={MODEL_NAME}", flush=True)

        tasks = ["task_1", "task_2", "task_3"] 
        for task_id in tasks:
            obs, info = env.reset(task_id=task_id)
            
            # API call through proxy
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Task: {task_id}. State: {obs}. Return JSON action."}],
                response_format={ "type": "json_object" }
            )
            
            content = json.loads(response.choices[0].message.content)
            action = Action(
                category=content.get("category", "FINANCE"),
                command=content.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=content.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            
            # Mandatory Step Log immediately after step
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            # Mandatory End Log
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
