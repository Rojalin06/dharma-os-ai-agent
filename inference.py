import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # Step 1: Environment variables ko Scaler format mein uthana
        # Default value wahi proxy URL hona chahiye
        base_url = os.getenv("API_BASE_URL", "https://proxy.llm.scaler.com/v1")
        api_key = os.getenv("HF_TOKEN") # Scaler checks this variable
        model_name = os.getenv("MODEL_NAME", "gpt-4o")

        if not api_key:
            print("[ERROR] HF_TOKEN is missing!")
            return

        # Step 2: Client initialization specifically via proxy
        client = OpenAI(base_url=base_url, api_key=api_key)
        env = DharmaEnv()

        print("[START] Dharma-OS Initialized")

        # Step 3: Tasks Execution
        tasks = ["task_1", "task_2", "task_3"]
        for task_id in tasks:
            obs, info = env.reset(task_id=task_id)

            # LLM Call
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"Analyze state: {obs} for task {task_id}. Respond in JSON."}],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            action = Action(
                category=data.get("category", "FINANCE"),
                command=data.get("command", "CANCEL_SUBSCRIPTION"),
                target_id=data.get("target_id", "Unknown")
            )

            obs, reward, done, info = await env.step(action)
            
            # Score scaling between 0.1 and 0.9
            final_reward = max(0.1, min(0.9, float(reward)))
            print(f"[STEP] Task: {task_id} | Reward: {final_reward}")

        print(f"[END] Final Score: {final_reward}")

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    asyncio.run(main())
