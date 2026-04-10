import asyncio
import os
import json
from openai import AsyncOpenAI
from env import DharmaEnv
from models import Action

async def run_task(client, env, task_id, model_name):
    print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)

    obs, info = env.reset(task_id=task_id)
    step_count = 0
    done = False
    total_reward = 0.0

    while not done and step_count < 10:
        prompt = f"""You are an AI agent managing DharmaOS.

Current State:
- Compliance Score: {obs.compliance_score}
- Active Subscriptions: {obs.active_subscriptions}
- Social Sentiment: {obs.social_sentiment}
- Pending Tasks: {obs.pending_tasks}

Priority order: LEGAL first, then FINANCE, then SOCIAL.

Reply ONLY in this exact JSON:
{{
  "category": "LEGAL",
  "command": "RESOLVE_COMPLIANCE",
  "target_id": "GDPR"
}}

Only use these categories: LEGAL, FINANCE, SOCIAL
Only use these commands: RESOLVE_COMPLIANCE, CANCEL_SUBSCRIPTION, HANDLE_COMPLAINT
For FINANCE, target_id must be one of: Slack, Adobe, Zoom"""

        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        content = json.loads(response.choices[0].message.content)

        action = Action(
            category=content.get("category", "LEGAL"),
            command=content.get("command", "RESOLVE_COMPLIANCE"),
            target_id=content.get("target_id", "Unknown")
        )

        obs, reward, done, info = await env.step(action)
        step_count += 1
        total_reward += reward

        print(f"[STEP] step={step_count} action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

    # ✅ Single float value — strictly (0.01, 0.99)
    final_score = max(0.01, min(total_reward, 0.99))

    # ✅ rewards= mein single value, list nahi
    print(f"[END] success={str(done).lower()} steps={step_count} score={final_score:.3f} rewards={final_score:.2f}", flush=True)

    return final_score


async def main():
    try:
        api_base   = os.environ["API_BASE_URL"]
        api_key    = os.environ["API_KEY"]
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        env = DharmaEnv()

        for task_id in ["task_1", "task_2", "task_3"]:
            await run_task(client, env, task_id, model_name)

    except KeyError as e:
        print(f"[CRITICAL ERROR] Variable {e} missing!", flush=True)
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
