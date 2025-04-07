import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import environment
from agent import Agent


class Hw3Env(environment.BaseEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._delta = 0.05
        self._goal_thresh = 0.075  # easier goal detection
        self._max_timesteps = 300  # allow more steps
        self._prev_obj_pos = None  # track object movement

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene
    
    def reset(self):
        super().reset()
        self._prev_obj_pos = self.data.body("obj1").xpos[:2].copy()  # initialize previous position
        self._t = 0

        try:
            return self.high_level_state()
        except:
            return None

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    # def reward(self):
    #     state = self.high_level_state()
    #     ee_pos = state[:2]
    #     obj_pos = state[2:4]
    #     goal_pos = state[4:6]
    #     ee_to_obj = max(10*np.linalg.norm(ee_pos - obj_pos), 1)
    #     obj_to_goal = max(10*np.linalg.norm(obj_pos - goal_pos), 1)
    #     goal_reward = 100 if self.is_terminal() else 0
    #     return 1/(ee_to_obj) + 1/(obj_to_goal) + goal_reward

    def reward(self):
        
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]

        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # distance-based rewards
        r_ee_to_obj = -0.1 * d_ee_to_obj  # getting closer to object
        r_obj_to_goal = -0.2 * d_obj_to_goal  # moving object to goal

        # direction bonus
        obj_movement = obj_pos - self._prev_obj_pos
        dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
        r_direction = 0.5 * max(0, np.dot(obj_movement / (np.linalg.norm(obj_movement) + 1e-8), dir_to_goal))
        if np.linalg.norm(obj_movement) < 1e-6:  # Avoid division by zero
            r_direction = 0.0

        # terminal bonus
        r_terminal = 10.0 if self.is_terminal() else 0.0

        r_step = -0.1  # penalty for each step

        self._prev_obj_pos = obj_pos.copy()
        return r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps
    
    def step(self, action):
        action = action.clamp(-1, 1).detach().cpu().numpy() * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        result = self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)            

        self._t += 1

        state = self.high_level_state()
        reward = self.reward()
        terminal = self.is_terminal()
        if result:  # If the action is successful
            truncated = self.is_truncated()
        else:  # If didn't realize the action
            truncated = True
        return state, reward, terminal, truncated


    # def step(self, action):
    #     action = action.clamp(-1, 1).cpu().numpy() * self._delta
    #     ee_pos = self.data.site(self._ee_site).xpos[:2]
    #     target_pos = np.concatenate([ee_pos, [1.06]])
    #     target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
    #     self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
    #     self._t += 1

    #     state = self.high_level_state()
    #     reward = self.reward()
    #     terminal = self.is_terminal()
    #     truncated = self.is_truncated()
    #     return state, reward, terminal, truncated

def train():
    env = Hw3Env(render_mode="offscreen")
    lr = 1e-5
    agent = Agent(learning_rate=lr)
    num_episodes = 10000

    window_size = 300  # Define the window size for the moving average

    rews = []
    mean_window_rewards = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 10))
    line1, = ax1.plot([], [], 'r-')  # Line for cumulative rewards
    line2, = ax2.plot([], [], 'b-')  # Line for mean window rewards

    # Configure the first subplot
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Training Progress - Cumulative Reward")
    ax1.set_xlim(0, num_episodes)
    ax1.set_ylim(-100, 100)
    ax1.grid()

    # Configure the second subplot
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mean Window Reward")
    ax2.set_title(f"Training Progress - Mean Reward (Window Size = {window_size})")
    ax2.set_xlim(0, num_episodes)
    ax2.set_ylim(-100, 100)
    ax2.grid()

    for i in range(num_episodes):        
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0

        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = agent.decide_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action[0])
            agent.add_reward(reward)
            cumulative_reward += reward
            done = is_terminal or is_truncated
            
            state = next_state
            episode_steps += 1

        print(f"Episode={i}, reward={cumulative_reward}")


        fault = agent.update_model()

        if fault:
            print("Fault detected, resetting memory.")
            continue

        rews.append(cumulative_reward)

        # Calculate the mean reward over the window
        if len(rews) >= window_size:
            mean_window_reward = np.mean(rews[-window_size:])
        else:
            mean_window_reward = np.mean(rews)
        mean_window_rewards.append(mean_window_reward)


        # Update the cumulative rewards plot
        line1.set_data(range(len(rews)), np.array(rews))
        ax1.relim()
        ax1.autoscale_view()

        # Update the mean window rewards plot
        line2.set_data(range(len(rews)), np.array(mean_window_rewards))
        ax2.relim()
        ax2.autoscale_view()

        plt.pause(0.1)


        # Save the plot every 100 episodes
        if i % 100 == 0:
            plt.savefig(f"training_progress_{lr}_{i}.png")
            print(f"Saved plot for episode {i}")
            torch.save(agent.model.state_dict(), f"model_{lr}_{i}.pt")
            print(f"Saved model for episode {i}")

    plt.savefig(f"training_progress_{lr}_final.png")
    print("Saved final plot")
    ## Save the model and the training statistics
    torch.save(agent.model.state_dict(), f"model_{lr}.pt")
    np.save("rews.npy", np.array(rews))

def test():
    env = Hw3Env(render_mode="gui")
    agent = Agent()
    agent.model.load_state_dict(torch.load("model_9500.pt"))
    num_episodes = 100

    rews = []

    for i in range(num_episodes):
        env.reset()

        state = env.high_level_state()
        
        done = False
        cumulative_reward = 0.0
        episode_steps = 0

        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = agent.decide_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action[0])
            cumulative_reward += reward
            done = is_terminal or is_truncated
            
            state = next_state
            episode_steps += 1

        print(f"Episode={i}, reward={cumulative_reward}")
        rews.append(cumulative_reward)

if __name__ == "__main__":
    test()  
    #train()

