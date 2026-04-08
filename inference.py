import asyncio
from env import DharmaEnv
from models import Action

async def main():
    env = DharmaEnv()
    obs = env.reset()
    print(f"[START] Dharma-OS Initialized. Compliance: {obs.compliance_score}")

    # Example Action: Finance Audit
    action = Action(category="FINANCE", command="CANCEL_SUBSCRIPTION", target_id="Adobe")
    obs, reward, done, info = await env.step(action)
    
    print(f"[STEP 1] Action: Cancel Adobe | Reward: {reward}")
    print(f"[END] Final Score: {obs.reward_score if hasattr(obs, 'reward_score') else reward}")

if __name__ == "__main__":
    asyncio.run(main())
