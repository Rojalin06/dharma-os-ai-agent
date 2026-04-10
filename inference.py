import asyncio
import os
import json
from openai import AsyncOpenAI
from env import DharmaEnv
from models import Action

async def main():
    api_base   = os.environ["API_BASE_URL"]
    api_key    = os.environ["API_KEY"]
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    tasks = ["task_1", "task_2", "task_3"]

    for task_id in tasks:
        env = DharmaEnv()  # ✅ Har task ke liye fresh env
        obs, info = env.reset(task_id=task_id)

        print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)

        step_count = 0
        done = False
        total_reward = 0.0

        while not done and step_count < 10:
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": f"""You manage DharmaOS. Reply ONLY valid JSON.
State: compliance={obs.compliance_score}, subs={list(obs.active_subscriptions.keys())}, sentiment={obs.social_sentiment}, pending={obs.pending_tasks}
Pick best action:
{{"category": "LEGAL", "command": "RESOLVE_COMPLIANCE", "target_id": "GDPR"}}
OR
{{"category": "FINANCE", "command": "CANCEL_SUBSCRIPTION", "target_id": "Slack"}}
OR  
{{"category": "SOCIAL", "command": "HANDLE_COMPLAINT", "target_id": "complaint"}}"""
                    }],
                    response_format={"type": "json_object"}
                )

                content = json.loads(response.choices[0].message.content)
                action = Action(
                    category=content.get("category", "LEGAL"),
                    command=content.get("command", "RESOLVE_COMPLIANCE"),
                    target_id=content.get("target_id", "GDPR")
                )

            except Exception:
                # Fallback action
                action = Action(category="LEGAL", command="RESOLVE_COMPLIANCE", target_id="GDPR")

            obs, reward, done, info = await env.step(action)
            step_count += 1
            total_reward += reward

            print(f"[STEP] step={step_count} action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

        # ✅ Strictly between 0.01 and 0.99
        final_score = round(max(0.01, min(total_reward, 0.99)), 3)

        print(f"[END] success={str(done).lower()} steps={step_count} score={final_score} rewards={final_score}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
