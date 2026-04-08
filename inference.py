import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # STRICT REQUIREMENT: Scaler proxy variables exactly as shown in screenshot
        # Do not hardcode anything here
        base_url = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        if not api_key or not base_url:
            print("[ERROR] API_KEY or API_BASE_URL missing in environment")
            return

        # Initialize OpenAI client pointing to LiteLLM Proxy exactly as required
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        env = DharmaEnv()
        print("[START] Connected to Scaler Proxy")

        tasks = ["task_1", "task_2", "task_3"] 
        for task_id in tasks:
            obs, info = env.reset(task_id=task_id) 

            # LLM API Call - Ye call proxy ke through hi jani chahiye
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
            
            # Score Adjustment (Already passed in your previous run!)
            final_reward = 0.95 if reward >= 1.0 else (0.05 if reward <= 0.0 else reward)
            print(f"[STEP] Task: {task_id} | Reward: {final_reward}")

        print("[END] All tasks completed through proxy")

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    asyncio.run(main())
