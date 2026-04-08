import asyncio
import os
import json
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        # STRICT FIX: Hum wahi variables use karenge jo Scaler/Meta inject kar raha hai
        # Unhone bataya hai: API_BASE_URL aur API_KEY
        api_key = os.environ.get("API_KEY") 
        base_url = os.environ.get("API_BASE_URL")
        model_name = os.environ.get("MODEL_NAME", "gpt-4o") # default agar missing ho

        if not api_key or not base_url:
            print("[ERROR] Missing Scaler Environment Variables (API_KEY or API_BASE_URL)")
            return

        # Initialize OpenAI client with THEIR proxy
        client = OpenAI(
            base_url=base_url, 
            api_key=api_key
        )

        env = DharmaEnv()
        tasks = ["task_1", "task_2", "task_3"] 
        
        print(f"[START] Dharma-OS Initialized via Proxy")

        for task_id in tasks:
            print(f"\n--- Processing {task_id} ---")
            obs, info = env.reset(task_id=task_id) 

            try:
                # API call through proxy
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user", 
                        "content": f"Task: {task_id}. State: {obs}. Return ONLY JSON with category, command, target_id."
                    }],
                    response_format={ "type": "json_object" }
                )
                
                raw_content = response.choices[0].message.content
                data = json.loads(raw_content)
                
                action = Action(
                    category=data.get("category", "FINANCE"),
                    command=data.get("command", "CANCEL_SUBSCRIPTION"),
                    target_id=data.get("target_id", "Unknown")
                )

                obs, reward, done, info = await env.step(action)
                
                # Reward adjustment logic (strictly between 0 and 1)
                final_reward = 0.95 if reward >= 1.0 else (0.05 if reward <= 0.0 else reward)
                print(f"[SUCCESS] Task: {task_id} | Final Score: {final_reward}")

            except Exception as task_err:
                print(f"[TASK ERROR] {task_id}: {task_err}")
                continue 

    except Exception as e:
        print(f"[CRITICAL ERROR] {str(e)}")
        return 

if __name__ == "__main__":
    asyncio.run(main())
