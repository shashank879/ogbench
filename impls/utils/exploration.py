from collections import defaultdict
import inspect
import tqdm

import jax
import numpy as np

from .evaluation import supply_rng


def collect_exploration_episodes(policy, env, num_episodes, config, temperature=1.0, gaussian_noise=None, max_steps=1000):
    """
    Collect exploration episodes using the current agent policy.

    Args:
        agent: The agent to use for exploration
        env: The environment
        num_episodes: Number of episodes to collect
        config: Agent config
        temperature: Temperature for action sampling
        gaussian_noise: Std of Gaussian noise to add to actions
        max_steps: Maximum steps per episode

    Returns:
        episodes: List of episode dicts (each in trajectory format)
        episode_returns: List of episode returns
        episode_lengths: List of episode lengths
    """
    actor_fn = supply_rng(policy, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    episodes = []
    episode_returns = []
    episode_lengths = []

    for ep in tqdm.tqdm(range(num_episodes), 'Exploring'):
        obs, info = env.reset()  # Get info too
        goal = info.get('goal')  # Get goal from info like in evaluation
        # first_obs = obs
        done = False
        episode_length = 0

        episode_data = defaultdict(list)
        episode_return = 0

        while not done and episode_length < max_steps:
            if episode_length % 100 == 0:
                first_obs = obs
            # Match evaluation code - no [None] indexing
            inputs = dict(observations=obs, goals=goal, temperature=temperature)
            if 'init_obs' in inspect.signature(policy).parameters:
                inputs['init_obs'] = first_obs
            action = actor_fn(**inputs)

            # Handle tuple return (action, action_info)
            if isinstance(action, tuple):
                action, _ = action
            action = np.array(action)

            if not config.get('discrete', False):   
                if gaussian_noise is not None:
                    action = np.random.normal(action, gaussian_noise)
                action = np.clip(action, -1, 1)

            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store in trajectory format
            episode_data['observations'].append(obs.copy())
            episode_data['actions'].append(action.copy() if isinstance(action, np.ndarray) else np.array([action]))
            # episode_data['rewards'].append(reward)
            episode_data['terminals'].append(float(terminated))
            episode_data['next_observations'].append(next_obs.copy())
            # if 'truncations' in env.unwrapped.__dict__ or truncated:
            #     episode_data['truncations'].append(float(truncated))

            episode_return += reward
            episode_length += 1
            obs = next_obs

        # Convert lists to arrays and set valids (compact format)
        episode_dict = {}
        for key, val_list in episode_data.items():
            episode_dict[key] = np.array(val_list)

        # Set valids: last state is invalid (can't be used as next_obs)
        episode_dict['valids'] = np.ones(episode_length, dtype=np.float32)
        episode_dict['valids'][-1] = 0.0

        # Ensure last terminal is set
        episode_dict['terminals'][-1] = 1.0

        episodes.append(episode_dict)
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    return episodes, episode_returns, episode_lengths
