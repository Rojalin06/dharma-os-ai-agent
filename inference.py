import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # STRICTLY as per Screenshot instructions
        # Use direct os.environ access so it fails if they aren't injected
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )
        
        # Use the provided model name
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        env = DharmaEnv()
        
        # Mandatory Start Log
        print(f"[START] task=task_1 env=dharma_os model={model_name}", flush=True)

        tasks = ["task_1", "task_2", "task_3"] 
        for task_id in tasks:
            obs, info = env.reset(task_id=task_id)
            
            # API Call that MUST hit the LiteLLM proxy
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
            
            # Strict formatting for logs
            print(f"[STEP] step=1 action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            print(f"[END] success={str(done).lower()} steps=1 score={reward:.3f} rewards={reward:.2f}", flush=True)

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
