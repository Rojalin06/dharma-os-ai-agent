import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # STRICT REQUIREMENT: Screenshot ke mutabiq exact variables
        # Default values hum yahan se hata rahe hain taaki validator proxy use kare
        base_url = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        if not api_key or not base_url:
            print(f"[ERROR] Missing Env Vars: API_KEY={bool(api_key)}, URL={bool(base_url)}")
            return

        # OpenAI Client initialization exactly as per "HOW TO FIX"
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        env = DharmaEnv()
        print("[START] Dharma-OS Initialized")

        # Running tasks
        tasks = ["task_1", "task_2", "task_3"] 
        for task_id in tasks:
            obs, info = env.reset(task_id=task_id) 

            # LLM API Call through Proxy
            response = client.chat.completions.create(
                model=model_name,
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
            
            # Score adjustment (Must be strictly between 0 and 1)
            final_reward = 0.95 if reward >= 1.0 else (0.05 if reward <= 0.0 else reward)
            print(f"[STEP] Task: {task_id} | Action: {action.command} | Reward: {final_reward}")

        print(f"[END] Final Score: {final_reward}")

    except Exception as e:
        print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
