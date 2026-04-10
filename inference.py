import asyncio
import os
import json
from openai import AsyncOpenAI
from env import DharmaEnv
from models import Action

async def run_task(client, env, task_id, model_name):
    # START line is fine
    print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)
    
    obs, info = env.reset(task_id=task_id)
    step_count = 0
    done = False
    reward_list = []

    while not done and step_count < 10:
        prompt = f"""You are an AI managing DharmaOS. 
State: Compliance={obs.compliance_score}, Subs={obs.active_subscriptions}, Sentiment={obs.social_sentiment}.
Return ONLY JSON: {{"category": "LEGAL/FINANCE/SOCIAL", "command": "STR", "target_id": "STR"}}"""

        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        content = json.loads(response.choices[0].message.content)
        action = Action(
            category=content.get("category", "LEGAL"),
            command=content.get("command", "RESOLVE"),
            target_id=content.get("target_id", "Unknown")
        )

        obs, reward, done, info = await env.step(action)
        step_count += 1
        reward_list.append(reward)

        # [STEP] line must be exact
        print(f"[STEP] step={step_count} action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

    # Calculate final score (Average) and clamp strictly between 0.01 and 0.99
    avg_reward = sum(reward_list) / len(reward_list) if reward_list else 0.01
    final_score = max(0.01, min(avg_reward, 0.99))
    
    # Format rewards as comma-separated string (e.g., "0.45,0.25")
    rewards_str = ",".join([f"{r:.2f}" for r in reward_list])

    # CRITICAL: Removed "task={task_id}" from [END] line to match validator regex
    print(f"[END] success={str(done).lower()} steps={step_count} score={final_score:.3f} rewards={rewards_str}", flush=True)
    
    return final_score

async def main():
    try:
        api_base = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        env = DharmaEnv()

        for task_id in ["task_1", "task_2", "task_3"]:
            await run_task(client, env, task_id, model_name)

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
