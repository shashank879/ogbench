import json
import os
import random
import time
import shutil
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset, DHPDataset
from utils.env_utils import make_env_and_datasets
from utils.eval_utils import visualize_goal_buffer_on_maze, create_goal_trajectory_video
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('exp_name', None, 'Experiment name.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
flags.DEFINE_boolean('debug', False, 'Run in debug mode.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = FLAGS.exp_name or get_exp_name(FLAGS.seed)
    if not FLAGS.debug:
        setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    else:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, 'debug', exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'DHPDataset': DHPDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)
        if hasattr(agent, 'non_jit_update'):
            non_jit_update_info = agent.non_jit_update(batch, i)
            update_info.update(non_jit_update_info)

        # Log metrics.
        if i==1 or i % FLAGS.log_interval == 0:
            if i > 1:
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(config['batch_size'])
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
                train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
                train_metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
            else:
                train_metrics = {}

            if hasattr(agent, 'goal_buffer'):
                # Add rendered goal positions as summary
                buffer_frame = visualize_goal_buffer_on_maze(
                    goal_buffer=agent.goal_buffer,
                    env=env,
                    render_size=1024,
                    goal_color=(0, 255, 0),  # Green
                    goal_radius=1,
                )
                train_metrics['buffer_goals'] = wandb.Image(buffer_frame)

                # Save to disk for debugging
                viz_dir = os.path.join(FLAGS.save_dir, 'goal_buffer')
                os.makedirs(viz_dir, exist_ok=True)
                Image.fromarray(buffer_frame).save(os.path.join(viz_dir, f'step_{i}.png'))

            if not FLAGS.debug:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            render_trajs = []
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders, cur_render_trajs = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                render_trajs.extend(cur_render_trajs)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                if not FLAGS.debug:
                    video = get_wandb_video(renders=renders.copy(), n_cols=num_tasks)
                    eval_metrics['video'] = video

                # Goal visualization video
                if hasattr(agent, 'goal_buffer') and len(render_trajs) > 0 and 'subgoals' in render_trajs[0]:
                    goal_video: wandb.Video = create_goal_trajectory_video(
                        render_trajs,
                        env,
                        renders=renders,
                        n_cols=num_tasks,
                        goal_buffer=agent.goal_buffer,
                    )
                    goal_viz_dir = os.path.join(FLAGS.save_dir, 'goal_video')
                    os.makedirs(goal_viz_dir, exist_ok=True)
                    shutil.copy(goal_video._path, os.path.join(goal_viz_dir, f'step_{i}.mp4'))
                    if goal_video is not None:
                        eval_metrics['subgoals_video'] = goal_video

            if not FLAGS.debug:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
