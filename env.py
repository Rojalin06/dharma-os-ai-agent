import asyncio
from models import Observation, Action

class DharmaEnv:
    def __init__(self):
        self.reset()

    def reset(self, task_id=None):
        self.task_id = task_id
        # Issues initialized
        self.subscriptions = {"Slack": 15.0, "Adobe": 50.0, "Zoom": 20.0}
        self.legal_issues = ["GDPR Section A missing"]
        self.social_alerts = ["Unresolved customer complaint"]
        self.steps = 0
        return self.get_state(), {}

    def get_state(self):
        return Observation(
            compliance_score=0.5 if self.legal_issues else 1.0,
            active_subscriptions=self.subscriptions,
            social_sentiment="Negative" if self.social_alerts else "Positive",
            pending_tasks=len(self.legal_issues) + len(self.social_alerts)
        )

    async def step(self, action: Action):
        self.steps += 1
        reward = 0.05 # Baseline reward for taking an action

        if action.category == "LEGAL" and self.legal_issues:
            self.legal_issues.pop()
            reward = 0.45
        elif action.category == "FINANCE" and action.target_id in self.subscriptions:
            del self.subscriptions[action.target_id]
            reward = 0.25
        elif action.category == "SOCIAL" and self.social_alerts:
            self.social_alerts.pop()
            reward = 0.15

        done = self.steps >= 5 or (not self.legal_issues and not self.social_alerts)
        
        # Strictly between 0 and 1 (Not 0.0 or 1.0)
        reward = max(0.01, min(reward, 0.99))

        return self.get_state(), reward, done, {}
