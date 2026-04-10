import asyncio
import os
import json
from openai import AsyncOpenAI
from env import DharmaEnv
from models import Action

async def main():
    try:
        api_base   = os.environ["API_BASE_URL"]
        api_key    = os.environ["API_KEY"]
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")

        client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        env = DharmaEnv()

        tasks = ["task_1", "task_2", "task_3"]

        for task_id in tasks:
            print(f"[START] task={task_id} env=dharma_os model={model_name}", flush=True)

            obs, info = env.reset(task_id=task_id)
            step_count = 0
            done = False
            last_reward = 0.05  # ✅ Default reward 0.0 nahi

            while not done:
                prompt = f"""You are an AI agent managing a company OS called DharmaOS.

Current State:
- Compliance Score: {obs.compliance_score}
- Active Subscriptions: {obs.active_subscriptions}
- Social Sentiment: {obs.social_sentiment}
- Pending Tasks: {obs.pending_tasks}

Your goal is to resolve all issues optimally.
Priority: LEGAL > FINANCE > SOCIAL

Reply ONLY in this exact JSON format (no extra text):
{{
  "category": "LEGAL" or "FINANCE" or "SOCIAL",
  "command": "RESOLVE_COMPLIANCE" or "CANCEL_SUBSCRIPTION" or "HANDLE_COMPLAINT",
  "target_id": "Slack" or "Adobe" or "Zoom" or "GDPR" or "complaint" or "Unknown"
}}"""

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
                last_reward = reward  # ✅ Last reward save karo

                print(f"[STEP] step={step_count} action={action.command} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

            # ✅ Final score strictly (0.0, 1.0) ke beech
            final_score = max(0.01, min(last_reward, 0.99))

            print(f"[END] success={str(done).lower()} steps={step_count} score={final_score:.3f} rewards={final_score:.2f}", flush=True)

    except KeyError as e:
        print(f"[CRITICAL ERROR] Variable {e} missing!", flush=True)
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
