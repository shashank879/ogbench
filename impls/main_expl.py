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
from utils.datasets import Dataset, GCDataset, HGCDataset, DHPDataset, DHPExplDataset, ReplayBuffer, MixedDataset
from utils.env_utils import make_env_and_datasets
from utils.eval_utils import visualize_goal_buffer_on_maze, create_goal_trajectory_video, plot_value_function_grid, compute_maze_coverage
from utils.evaluation import evaluate
from utils.exploration import collect_exploration_episodes
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
flags.DEFINE_string('best_metric_key', 'evaluation/overall_success', 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_integer('expl_eval_episodes', 10, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
flags.DEFINE_boolean('debug', False, 'Run in debug mode.')

# Exploration flags
flags.DEFINE_enum('expl_mode', 'none', ['none', 'pretrain', 'interleaved'], 'Exploration mode: none (offline only), pretrain (explore then train), interleaved (explore during training)')
flags.DEFINE_integer('expl_episodes', 10, 'Number of exploration episodes (for pretrain mode)')
flags.DEFINE_integer('expl_interval', 5000, 'Exploration interval in training steps (for interleaved mode)')
flags.DEFINE_integer('replay_buffer_size', 1000000, 'Size of online replay buffer')
flags.DEFINE_float('offline_online_ratio', 0.5, 'Ratio of offline to online data (0.5 = 50% offline, 50% online). 1.0 = only offline, 0.0 = only online')
flags.DEFINE_float('expl_temperature', 1.0, 'Temperature for exploration policy')
flags.DEFINE_float('expl_gaussian', None, 'Gaussian noise std for exploration')
flags.DEFINE_integer('expl_max_steps', 1000, 'Max steps per exploration episode')
flags.DEFINE_boolean('recency_sampling', True, 'Use recency sampling for the replay buffer')
flags.DEFINE_string('recency_strategy', 'windowed_exp', 'Recency sampling strategy: [windowed_exp, exp, rank, power]')

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
    print('[SAVE DIR] : ', FLAGS.save_dir)
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
        'DHPExplDataset': DHPExplDataset,
    }[config['dataset_class']]

    # Create offline dataset wrapper
    offline_dataset: Dataset = Dataset.create(**train_dataset)
    print('Offline dataset size: ', offline_dataset.size)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    if FLAGS.expl_mode != 'none' and FLAGS.offline_online_ratio < 1.:
        example_transition = {k: v[0] for k,v in offline_dataset.sample(1).items()}
        replay_buffer = ReplayBuffer.create(example_transition, FLAGS.replay_buffer_size, use_recency=FLAGS.recency_sampling, recency_strat=FLAGS.recency_strategy)
        print(f'[REPLAY BUFFER] Created empty buffer of capacity {FLAGS.replay_buffer_size}, and current size {replay_buffer.size}')

        if FLAGS.offline_online_ratio == .0:
            train_datastore = replay_buffer
            print('Using only replay buffer')
        else:
            train_datastore = MixedDataset.create(
                datasets=[offline_dataset, replay_buffer],
                ratios=[FLAGS.offline_online_ratio, 1.0 - FLAGS.offline_online_ratio]
            )
            print(f'[DATASET] Using mixed dataset: {FLAGS.offline_online_ratio:.1%} offline, {1-FLAGS.offline_online_ratio:.1%} online')
    else:
        replay_buffer = None
        train_datastore = offline_dataset
        print(f'[DATASET] Using 100% offline data')

    # Create GCDataset wrapper once - it will auto-update when underlying data changes
    train_dataset: GCDataset = dataset_class(train_datastore, config)
    print(f'[DATASET] Initial size: {train_dataset.size}')

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    example_batch = dataset_class(offline_dataset, config).sample(1)
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
    best_metric = None
    total_expl_episodes = 0
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)

    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True, desc='Training'):
        # INTERLEAVED EXPLORATION: Collect data during training
        if FLAGS.expl_mode != 'none' and (i==1 or i % FLAGS.expl_interval == 0):
            episodes, _, _ = collect_exploration_episodes(
                policy=agent.explore,
                env=env,
                num_episodes=FLAGS.expl_episodes,
                config=config,
                temperature=FLAGS.expl_temperature,
                gaussian_noise=FLAGS.expl_gaussian,
                max_steps=FLAGS.expl_max_steps
            )

            if replay_buffer:
                # Add episodes to buffer
                for episode in episodes:
                    replay_buffer.add_episode(episode, i)

                total_expl_episodes += FLAGS.expl_episodes

                recency_stats = replay_buffer.get_recency_stats()
                buffer_metrics = {
                    'buffer/train_dataset_size': train_dataset.size,  # This will show the updated size,
                    'buffer/offline_size': offline_dataset.size,
                    'buffer/online_size': replay_buffer.size,
                    'buffer/num_collection_steps': recency_stats['num_collection_steps'],
                    'buffer/train_step_range': recency_stats['train_step_range'],
                    'buffer/oldest_train_step': recency_stats['oldest_train_step'],
                }
                if not FLAGS.debug:
                    wandb.log(buffer_metrics, step=i)
                train_logger.log(buffer_metrics, step=i)

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

            if hasattr(env.unwrapped, 'maze_map') and len(batch['observations'].shape) == 2 and hasattr(agent, 'goal_buffer'):
                # Add rendered goal positions as summary
                buffer_frame = visualize_goal_buffer_on_maze(
                    goal_buffer=agent.goal_buffer,
                    env=env,
                    render_size=1024,
                    value_fn=lambda s,g: agent.network.select(config['high_act_val_fn'])(s,g).mean(0)
                )
                train_metrics['buffer_goals'] = wandb.Image(buffer_frame)

                # Save to disk for debugging
                viz_dir = os.path.join(FLAGS.save_dir, 'goal_buffer')
                os.makedirs(viz_dir, exist_ok=True)
                Image.fromarray(buffer_frame).save(os.path.join(viz_dir, f'step_{i}.png'))

            val_image = plot_value_function_grid(
                agent=agent,
                agent_name = config['agent_name'],
                n_tasks=num_tasks,
                env=env,
                grid_size=100,
                output_path=os.path.join(FLAGS.save_dir, 'value_func_image', f'step_{i}.png')
            )
            train_metrics['val_image'] = wandb.Image(val_image)

            if not FLAGS.debug:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent

            # Plan evaluation
            print('Evaluating Planner...')
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            all_trajs = []
            render_trajs = []
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders, cur_render_trajs = evaluate(
                    policy=eval_agent.sample_actions,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                all_trajs.append(trajs)
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
            
            val_image = plot_value_function_grid(
                agent=agent,
                agent_name = config['agent_name'],
                n_tasks=num_tasks,
                env=env,
                grid_size=100,
                output_path=os.path.join(FLAGS.save_dir, 'plan_trajs', f'step_{i}.png'),
                all_trajs=all_trajs,
            )
            eval_metrics['plan_trajs'] = wandb.Image(val_image)

            if FLAGS.video_episodes > 0:
                if not FLAGS.debug:
                    video = get_wandb_video(renders=renders.copy(), n_cols=num_tasks)
                    eval_metrics['video'] = video

                # Goal visualization video
                if hasattr(env.unwrapped, 'maze_map') and hasattr(agent, 'goal_buffer') and len(render_trajs) > 0 and 'subgoals' in render_trajs[0]:
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

            # Explorer evaluation
            print('Evaluating Explorer...')
            renders = []
            # expl_metrics = {}
            overall_metrics = defaultdict(list)
            all_trajs = []
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                _, trajs, cur_renders, cur_render_trajs = evaluate(
                    policy=eval_agent.explore,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.expl_eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.expl_temperature,
                    eval_gaussian=FLAGS.expl_gaussian,
                )
                all_trajs.append(trajs)
                renders.extend(cur_renders)

            coverage_metrics = compute_maze_coverage(env, all_trajs)
            eval_metrics.update({'expl_evaluation/' + k:v for k,v in coverage_metrics.items()})

            val_image = plot_value_function_grid(
                agent=agent,
                agent_name = config['agent_name'],
                n_tasks=num_tasks,
                env=env,
                grid_size=100,
                output_path=os.path.join(FLAGS.save_dir, 'expl_trajs', f'step_{i}.png'),
                all_trajs=all_trajs,
            )
            eval_metrics['expl_trajs'] = wandb.Image(val_image)

            if FLAGS.video_episodes > 0:
                if not FLAGS.debug:
                    video = get_wandb_video(renders=renders.copy(), n_cols=num_tasks)
                    eval_metrics['expl_video'] = video

            if not FLAGS.debug:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

            if (best_metric is None) or (eval_metrics['evaluation/overall_success'] >= best_metric):
                save_agent(agent, FLAGS.save_dir, 'best')
                best_metric = eval_metrics['evaluation/overall_success']

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
