import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # STRICTLY as per Scaler Checklist/Video
        # Use exact names: API_BASE_URL and API_KEY
        base_url = os.environ.get("API_BASE_URL", "https://proxy.llm.scaler.com/v1")
        api_key = os.environ.get("API_KEY") 
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        if not api_key:
            print("[ERROR] API_KEY missing in environment")
            return

        # Initializing client via Proxy
        client = OpenAI(base_url=base_url, api_key=api_key)
        env = DharmaEnv()

        print("[START] Dharma-OS Initialized")

        tasks = ["task_1", "task_2", "task_3"]
        for task_id in tasks:
            obs, info = env.reset(task_id=task_id)

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"Task: {task_id}. State: {obs}. Return JSON."}],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            action = Action(
                category=data.get("category", "FINANCE"),
                command=data.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=data.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            
            # Score strictly between 0 and 1
            final_reward = 0.95 if reward >= 1.0 else (0.05 if reward <= 0.0 else reward)
            print(f"[STEP] Task: {task_id} | Action: {action.command} | Reward: {final_reward}")

        print(f"[END] Final Score: {final_reward}")

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    asyncio.run(main())
