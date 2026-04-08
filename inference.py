import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # 1. Variables ko direct access karein
        # Agar ye Hugging Face mein nahi milenge, toh code crash hoga aur humein pata chal jayega
        api_key = os.environ["API_KEY"] 
        base_url = os.environ["API_BASE_URL"]
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        # 2. Print karke verify karein (Sirf debugging ke liye, key print mat karna)
        print(f"[DEBUG] Using Base URL: {base_url}")

        client = OpenAI(
            base_url=base_url, 
            api_key=api_key
        )

        env = DharmaEnv()
        # Ensure teeno tasks register ho rahe hain
        tasks = ["task_1", "task_2", "task_3"] 
        
        for task_id in tasks:
            obs, info = env.reset(task_id=task_id) 

            # LLM API Call
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
            
            # Score Adjustment
            final_reward = 0.95 if reward >= 1.0 else (0.05 if reward <= 0.0 else reward)
            print(f"Task: {task_id} | Score: {final_reward}")

    except KeyError as e:
        print(f"[CRITICAL] Variable Missing in Hugging Face: {e}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    asyncio.run(main())
