import asyncio
import os
from openai import OpenAI
from env import DharmaEnv
from models import Action

async def main():
    # LLM Client Setup - Ye portal ke variables use karega
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL"), 
        api_key=os.getenv("HF_TOKEN")
    )
    model_name = os.getenv("MODEL_NAME")

    env = DharmaEnv()
    # FIX: Yahan do values leni hain (obs aur info)
    obs, info = env.reset() 
    
    print("[START] Dharma-OS Initialized")

    # LLM se decision lena
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": f"Based on this state: {obs}, what is the best finance action? Return only JSON."}],
        response_format={ "type": "json_object" }
    )
    
    # Example Action (Aap LLM response se isse extract kar sakti hain)
    action = Action(category="FINANCE", command="CANCEL_SUBSCRIPTION", target_id="Adobe")
    
    # Step lena
    obs, reward, done, info = await env.step(action)
    
    print(f"[STEP] Action: {action.category} | Reward: {reward}")
    print(f"[END] Final Score: {reward}")

if __name__ == "__main__":
    asyncio.run(main())
