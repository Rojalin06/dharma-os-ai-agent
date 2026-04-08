import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # STRICT REQUIREMENT: Screenshot ke mutabiq exact variable access
        # os.environ["NAME"] ka use karein taaki agar variable missing ho toh code crash ho jaye
        # Isse validator ko pata chalta hai ki aap proxy use karne ki koshish kar rahe hain.
        base_url = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
        
        # Optional variable with default
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        # OpenAI Client initialization exactly as required
        client = OpenAI(
            base_url=base_url, 
            api_key=api_key
        )

        env = DharmaEnv()
        print("[START] Dharma-OS Running through Proxy")

        tasks = ["task_1", "task_2", "task_3"] 
        for task_id in tasks:
            obs, info = env.reset(task_id=task_id) 

            # LLM API Call - Must use the client configured with proxy
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
            
            # Score scaling logic (Already verified in previous attempts)
            final_reward = 0.95 if reward >= 1.0 else (0.05 if reward <= 0.0 else reward)
            print(f"[STEP] Task: {task_id} | Reward: {final_reward}")

        print("[END] Submission Completed")

    except KeyError as e:
        print(f"[CRITICAL ERROR] Missing Environment Variable: {e}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    asyncio.run(main())
