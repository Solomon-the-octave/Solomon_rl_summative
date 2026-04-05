import numpy as np
import gymnasium as gym
from gymnasium import spaces


class UrbanPlanningEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=30):
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(8,),
            dtype=np.float32
        )

        self.action_names = {
            0: "Do nothing",
            1: "Build housing",
            2: "Improve transport",
            3: "Expand water and sanitation",
            4: "Upgrade informal settlements",
            5: "Invest in green space",
            6: "Strengthen essential services"
        }

        self.state = None
        self.last_reward = 0.0
        self.last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array([
            60,   # population pressure
            45,   # housing
            40,   # transport
            50,   # water and sanitation
            35,   # green space
            55,   # informal settlements
            50,   # pollution
            100   # budget
        ], dtype=np.float32)

        self.current_step = 0
        self.last_reward = 0.0
        self.last_action = None

        return self.state.copy(), {}

    def step(self, action):
        action = int(action)
        self.last_action = action

        population, housing, transport, water, green, informal, pollution, budget = self.state

        if action == 0:
            population += 3
            informal += 2
            pollution += 2

        elif action == 1:
            housing += 12
            informal -= 8
            budget -= 15
            pollution += 2

        elif action == 2:
            transport += 12
            pollution -= 6
            population -= 3
            budget -= 15

        elif action == 3:
            water += 12
            informal -= 4
            budget -= 12

        elif action == 4:
            informal -= 12
            housing += 5
            water += 5
            budget -= 18

        elif action == 5:
            green += 12
            pollution -= 8
            budget -= 10

        elif action == 6:
            water += 4
            population -= 2
            informal -= 3
            budget -= 10

        population += 2

        if housing < 40:
            informal += 2

        if transport < 40:
            pollution += 2

        population = np.clip(population, 0, 100)
        housing = np.clip(housing, 0, 100)
        transport = np.clip(transport, 0, 100)
        water = np.clip(water, 0, 100)
        green = np.clip(green, 0, 100)
        informal = np.clip(informal, 0, 100)
        pollution = np.clip(pollution, 0, 100)
        budget = np.clip(budget, 0, 100)

        self.state = np.array([
            population,
            housing,
            transport,
            water,
            green,
            informal,
            pollution,
            budget
        ], dtype=np.float32)

        reward = (
            0.15 * housing
            + 0.15 * transport
            + 0.15 * water
            + 0.10 * green
            - 0.15 * population
            - 0.15 * informal
            - 0.15 * pollution
        )

        terminated = False
        truncated = False

        success = (
            housing >= 70 and
            transport >= 65 and
            water >= 65 and
            green >= 50 and
            informal <= 30 and
            pollution <= 35
        )

        failure = (
            budget <= 0 or
            pollution >= 90 or
            informal >= 90 or
            population >= 95
        )

        if success:
            reward += 20
            terminated = True

        if failure:
            reward -= 20
            terminated = True

        self.current_step += 1

        if self.current_step >= self.max_steps:
            truncated = True

        self.last_reward = float(reward)

        info = {
            "success": success,
            "failure": failure
        }

        return self.state.copy(), float(reward), terminated, truncated, info

    def get_state(self):
        return self.state.copy()

    def get_metrics(self):
        labels = [
            "Population Pressure",
            "Housing",
            "Transport",
            "Water and Sanitation",
            "Green Space",
            "Informal Settlements",
            "Pollution",
            "Budget"
        ]
        return dict(zip(labels, self.state.tolist()))