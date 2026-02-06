import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from an example transition.

        Args:
            transition: Example transition (dict with scalar values).
            size: Maximum size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)

        # Create instance without calling Dataset.__init__ yet
        instance = cls.__new__(cls)
        FrozenDict.__init__(instance, buffer_dict)
        instance.max_size = size
        instance.size = 0
        instance.train_steps = np.zeros(size, dtype=np.int64)
        instance.use_recency = True
        instance.recency_strategy = 'windowed_exp'
        instance.recency_alpha = 1e-5
        instance.recency_window = 50000

        # Set valid_idxs if valids key exists
        if 'valids' in buffer_dict:
            instance.valid_idxs = np.array([], dtype=np.int64)

        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = 0
        self.train_steps = None
        self.use_recency = False
        self.recency_strategy = 'windowed_exp'
        self.recency_alpha = 1e-5
        self.recency_window = 50000

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            init_size = min(len(init_buffer), size)
            buffer[:init_size] = init_buffer[:init_size]
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)

        instance = cls.__new__(cls)
        FrozenDict.__init__(instance, buffer_dict)
        instance.max_size = size
        instance.size = min(get_size(init_dataset), size)
        instance.train_steps = np.zeros(size, dtype=np.int64)
        instance.use_recency = True
        instance.recency_strategy = 'windowed_exp'
        instance.recency_alpha = 1e-5
        instance.recency_window = 50000

        # Set valid_idxs if valids exist
        if 'valids' in buffer_dict:
            valids = buffer_dict['valids'][:instance.size]
            (instance.valid_idxs,) = np.nonzero(valids > 0)

        return instance

    def add_transition(self, transition, train_step=0):
        """Add a single transition to the buffer."""
        if self.size < self.max_size:
            # Buffer not full, just append
            idx = self.size
            self.size += 1
        else:
            # Buffer full, shift everything left by 1 and add at end
            for key in self._dict.keys():
                self._dict[key][:-1] = self._dict[key][1:]
            self.train_steps[:-1] = self.train_steps[1:]
            idx = self.size - 1

        # Write new transition at idx
        for key, value in transition.items():
            self._dict[key][idx] = value
        self.train_steps[idx] = train_step

        # Update valid_idxs if needed
        if 'valids' in self._dict:
            valids = self._dict['valids'][:self.size]
            (self.valid_idxs,) = np.nonzero(valids > 0)

    def add_transitions(self, transitions, train_step=0):
        """Add multiple transitions to the buffer efficiently."""
        num_new = len(transitions)
        if num_new == 0:
            return

        if self.size + num_new <= self.max_size:
            # All transitions fit without shifting
            for i, transition in enumerate(transitions):
                idx = self.size + i
                for key, value in transition.items():
                    self._dict[key][idx] = value
                self.train_steps[idx] = train_step
            self.size += num_new
        else:
            # Need to shift or replace
            if num_new >= self.max_size:
                # New data fills entire buffer, just take last max_size transitions
                start_idx = num_new - self.max_size
                for i, transition in enumerate(transitions[start_idx:]):
                    for key, value in transition.items():
                        self._dict[key][i] = value
                    self.train_steps[i] = train_step
                self.size = self.max_size
            else:
                # Shift existing data and add new
                overflow = (self.size + num_new) - self.max_size
                # Shift left by overflow
                for key in self._dict.keys():
                    self._dict[key][:-overflow] = self._dict[key][overflow:]
                self.train_steps[:-overflow] = self.train_steps[overflow:]
                # Add new transitions at end
                for i, transition in enumerate(transitions):
                    idx = self.size - overflow + i
                    for key, value in transition.items():
                        self._dict[key][idx] = value
                    self.train_steps[idx] = train_step
                self.size = self.max_size

        # Update valid_idxs if needed
        if 'valids' in self._dict:
            valids = self._dict['valids'][:self.size]
            (self.valid_idxs,) = np.nonzero(valids > 0)

    def add_episode(self, episode_dict, train_step=0):
        """Add a complete episode to the buffer."""
        episode_length = len(episode_dict['observations'])
        transitions = []
        for i in range(episode_length):
            transition = {key: val[i] for key, val in episode_dict.items()}
            transitions.append(transition)
        self.add_transitions(transitions, train_step=train_step)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices.

        If use_recency is True, samples with bias toward recent training steps.
        Otherwise, samples uniformly.

        Args:
            num_idxs: Number of indices to sample

        Returns:
            Array of sampled indices
        """
        if 'valids' in self._dict:
            valid_positions = self.valid_idxs
        else:
            valid_positions = np.arange(self.size)

        if not self.use_recency or self.size == 0:
            # Uniform sampling (original behavior)
            return valid_positions[np.random.randint(len(valid_positions), size=num_idxs)]

        # Recency-weighted sampling
        # Get train steps for valid positions
        train_steps_valid = self.train_steps[valid_positions]

        # Find unique train steps and their inverse mapping
        unique_steps, inverse_indices = np.unique(train_steps_valid, return_inverse=True)

        # Compute weights based on recency of train steps
        # Higher train_step = more recent = higher weight
        if len(unique_steps) > 1:
            if self.recency_strategy == 'windowed_exp':
                # Windowed exponential decay
                max_step = unique_steps.max()
                ages = max_step - unique_steps
                step_weights = np.where(
                    ages <= self.recency_window,
                    1.0,
                    np.exp(-self.recency_alpha * (ages - self.recency_window))
                )

            elif self.recency_strategy == 'exp':
                # Pure exponential decay
                max_step = unique_steps.max()
                ages = max_step - unique_steps
                step_weights = np.exp(-self.recency_alpha * ages)

            elif self.recency_strategy == 'rank':
                # Rank-based (newest = highest rank)
                ranks = np.arange(1, len(unique_steps) + 1)
                step_weights = ranks ** self.recency_alpha

            elif self.recency_strategy == 'power':
                # Power of normalized step number (original approach, but normalized)
                min_step = unique_steps.min()
                normalized_steps = unique_steps - min_step + 1
                step_weights = normalized_steps ** self.recency_alpha

            else:
                raise ValueError(f"Unknown recency strategy: {self.recency_strategy}")
        else:
            step_weights = np.ones(len(unique_steps))

        # Map step weights back to individual transitions
        transition_weights = step_weights[inverse_indices]
        transition_weights = transition_weights / transition_weights.sum()

        # Sample with replacement according to weights
        sampled_positions = np.random.choice(
            len(valid_positions),
            size=num_idxs,
            replace=True,
            p=transition_weights
        )
        return valid_positions[sampled_positions]

    def clear(self):
        """Clear the replay buffer."""
        self.size = 0
        if 'valids' in self._dict:
            self.valid_idxs = np.array([], dtype=np.int64)

    def get_recency_stats(self):
        """Get statistics about training step recency in the buffer.

        Useful for monitoring/debugging.
        """
        if self.size == 0:
            return {}

        valid_positions = self.valid_idxs if 'valids' in self._dict else np.arange(self.size)
        train_steps_valid = self.train_steps[valid_positions]
        unique_steps = np.unique(train_steps_valid)

        return {
            'num_collection_steps': len(unique_steps),
            'oldest_train_step': int(unique_steps.min()),
            'newest_train_step': int(unique_steps.max()),
            'train_step_range': int(unique_steps.max() - unique_steps.min()),
        }


class MixedDataset:
    """
    Optimized Dataset that mixes multiple datasets with specified ratios.
    """

    @classmethod
    def create(cls, datasets: list[Dataset], ratios: list[float], freeze=True):
        """Create a mixed dataset from multiple Dataset objects."""
        assert len(datasets) == len(ratios), "Number of datasets must match number of ratios"
        assert np.isclose(sum(ratios), 1.0), f"Ratios must sum to 1.0, got {sum(ratios)}"
        assert all(r >= 0 for r in ratios), "All ratios must be non-negative"
        assert len(datasets) > 0, "Must provide at least one dataset"

        # Get keys from first dataset
        first_ds = datasets[0]
        initial_dict = {key: np.array([]) for key in first_ds._dict.keys()}

        # Create instance
        instance = super(MixedDataset, cls).__new__(cls)
        FrozenDict.__init__(instance, initial_dict)

        # Store attributes
        instance.datasets = datasets
        instance.ratios = ratios
        instance.num_datasets = len(datasets)
        instance._cache = {}
        instance._cached_size = 0
        instance._cached_cumsum = None  # NEW: Cache cumulative sizes

        # Compute initial size
        instance._size = sum(ds.size for ds in datasets)

        # Compute valid_idxs if any dataset has valids
        if any('valids' in ds._dict for ds in datasets):
            instance._recompute_valid_idxs()
        else:
            instance.valid_idxs = None

        return instance

    def _recompute_valid_idxs(self):
        """Recompute valid indices from all datasets."""
        valid_idxs_list = []
        cumsum = self._get_cumulative_sizes()  # Use cached version

        for i, ds in enumerate(self.datasets):
            offset = cumsum[i]
            if hasattr(ds, 'valid_idxs') and ds.valid_idxs is not None:
                valid_idxs_list.append(ds.valid_idxs + offset)
            elif 'valids' in ds._dict:
                (local_valid,) = np.nonzero(ds['valids'][:ds.size] > 0)
                valid_idxs_list.append(local_valid + offset)
            else:
                valid_idxs_list.append(np.arange(ds.size) + offset)

        self.valid_idxs = np.concatenate(valid_idxs_list) if valid_idxs_list else None

    def _get_cumulative_sizes(self):
        """Get cumulative sizes with caching."""
        current_size = sum(ds.size for ds in self.datasets)

        # Invalidate cache if size changed
        if self._cached_cumsum is None or current_size != self._cached_size:
            sizes = [ds.size for ds in self.datasets]
            self._cached_cumsum = np.cumsum([0] + sizes)
            self._cached_size = current_size

        return self._cached_cumsum

    def __getitem__(self, key):
        """Override to provide lazy concatenation with caching."""
        current_size = self.size

        # Invalidate cache if dataset size changed
        if current_size != self._cached_size:
            self._cache.clear()
            self._cached_cumsum = None  # Invalidate cumsum cache
            self._cached_size = current_size

            # Recompute valid_idxs if valids exist
            if any('valids' in ds._dict for ds in self.datasets):
                self._recompute_valid_idxs()

        # Return cached if available
        if key in self._cache:
            return self._cache[key]

        # Concatenate from all datasets
        arrays = []
        for ds in self.datasets:
            if ds.size > 0:
                arrays.append(ds._dict[key][:ds.size])

        if not arrays:
            raise KeyError(f"Key '{key}' not found or all datasets empty")

        result = np.concatenate(arrays)
        self._cache[key] = result
        return result

    @property
    def size(self):
        """Dynamically compute total size from current dataset sizes."""
        self._size = sum(ds.size for ds in self.datasets)
        return self._size

    @property
    def dataset_sizes(self):
        """Dynamically get current sizes of all datasets."""
        return [ds.size for ds in self.datasets]

    @property
    def cumulative_sizes(self):
        """Dynamically compute cumulative sizes (with caching)."""
        return self._get_cumulative_sizes()

    def _global_to_local_idx_vectorized(self, global_idxs):
        """
        Convert global indices to (dataset_ids, local_idxs) - VECTORIZED.

        This is the key optimization: processes all indices at once.
        """
        cumsum = self._get_cumulative_sizes()

        # Vectorized searchsorted - finds which dataset each index belongs to
        dataset_ids = np.searchsorted(cumsum[1:], global_idxs, side='right')

        # Vectorized subtraction to get local indices
        local_idxs = global_idxs - cumsum[dataset_ids]

        return dataset_ids, local_idxs

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices respecting mixing ratios."""
        idxs_list = []
        cumsum = self._get_cumulative_sizes()

        for i, (ds, ratio) in enumerate(zip(self.datasets, self.ratios)):
            n_samples = int(num_idxs * ratio)

            # Adjust last dataset to ensure exact num_idxs
            if i == self.num_datasets - 1:
                n_samples = num_idxs - sum(len(idx) for idx in idxs_list)

            if n_samples > 0 and ds.size > 0:
                local_idxs = ds.get_random_idxs(n_samples)
                global_idxs = local_idxs + cumsum[i]
                idxs_list.append(global_idxs)

        return np.concatenate(idxs_list) if idxs_list else np.array([], dtype=np.int64)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """
        Return a subset of the dataset given the indices.

        OPTIMIZED: Uses vectorized operations instead of Python loops.
        """
        # Vectorized index mapping - processes all indices at once!
        dataset_ids, local_idxs = self._global_to_local_idx_vectorized(idxs)

        # Pre-allocate result dictionary
        result = None

        # Process each dataset
        for dataset_id in range(self.num_datasets):
            # Find all indices belonging to this dataset
            mask = dataset_ids == dataset_id

            if not np.any(mask):
                continue

            # Get positions in output array
            positions = np.where(mask)[0]
            dataset_local_idxs = local_idxs[mask]

            # Sample from this dataset
            batch = self.datasets[dataset_id].get_subset(dataset_local_idxs)

            # Initialize result on first batch
            if result is None:
                result = {}
                for key, val in batch.items():
                    shape = (len(idxs),) + val.shape[1:]
                    result[key] = np.zeros(shape, dtype=val.dtype)

            # Place batch data in correct positions (vectorized assignment)
            for key, val in batch.items():
                result[key][positions] = val

        return result


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        goal_idxs = np.where(
            np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6), middle_goal_idxs, random_goal_idxs
        )

        # Goals at the current state.
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                )

        return batch


class DHPDataset(HGCDataset):

    def sample_low_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, max_distance=None):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if max_distance:
            final_state_idxs = np.minimum(idxs + max_distance, final_state_idxs)
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        goal_idxs = np.where(
            np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6), middle_goal_idxs, random_goal_idxs
        )

        # Goals at the current state.
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def sample_high_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, normal_subgoal_sample):
        batch_size = len(idxs)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        distances = np.random.rand(batch_size)  # in [0, 1)
        value_goal_idxs = np.round(
            (np.minimum(idxs + 2, final_state_idxs) * distances +
            final_state_idxs * (1 - distances))
        ).astype(int)

        if normal_subgoal_sample:
            subgoal_distances = np.clip(np.random.normal(0.5, 0.5, batch_size), 0, 1)
        else:
            subgoal_distances = np.random.rand(batch_size)  # in [0, 1)

        value_subgoal_idxs = np.round(
            (np.minimum(idxs, value_goal_idxs) * subgoal_distances + value_goal_idxs * (1 - subgoal_distances))
        ).astype(int)

        pick_random = np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6)
        value_goal_idxs = np.where(pick_random, value_goal_idxs, self.dataset.get_random_idxs(batch_size))
        value_subgoal_idxs = np.where(pick_random, value_subgoal_idxs, self.dataset.get_random_idxs(batch_size))

        value_goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, value_goal_idxs)
        value_subgoal_idxs = np.where((value_goal_idxs == idxs) * (np.random.rand(batch_size) < .7), idxs, value_subgoal_idxs)

        return value_goal_idxs, value_subgoal_idxs

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # Sample low value goals.
        low_value_goal_idxs = self.sample_low_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
            self.config.get('value_max_dist', None),
        )
        batch['low_value_goals'] = self.get_observations(low_value_goal_idxs)

        low_successes = (idxs == low_value_goal_idxs).astype(float)
        batch['low_masks'] = 1.0 - low_successes
        batch['low_rewards'] = low_successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Sample high value goals.
        value_goal_idxs, value_subgoal_idxs = self.sample_high_goals(
            idxs,
            self.config['high_value_p_curgoal'],
            self.config['high_value_p_trajgoal'],
            self.config['high_value_p_randomgoal'],
            self.config['high_value_normal_subg_sample'],
        )

        batch['high_value_goals'] = self.get_observations(value_goal_idxs)
        batch['high_value_subgoals'] = self.get_observations(value_subgoal_idxs)

        step_dist = self.config.get('high_value_min_dist', self.config['subgoal_steps'])
        successes_left = (np.abs(idxs - value_subgoal_idxs) <= step_dist).astype(float)
        successes_right = (np.abs(value_subgoal_idxs - value_goal_idxs) <= step_dist).astype(float)
        batch['masks_left'] = 1.0 - successes_left
        batch['masks_right'] = 1.0 - successes_right
        batch['rewards_left'] = successes_left - (1.0 if self.config['gc_negative'] else 0.0)
        batch['rewards_right'] = successes_right - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low-level actor goals.
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if self.config['hierarchical_planner']:
            high_traj_target_idxs = np.minimum(
                (idxs + high_traj_goal_idxs) // 2,
                high_traj_goal_idxs)
        else:
            high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        if self.config['hierarchical_planner']:
            high_random_target_idxs = np.minimum(
                (idxs + high_traj_goal_idxs) // 2,
                final_state_idxs)
        else:
            high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)
        batch['start_obs'] = self.get_observations(self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1])

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'low_value_goals',
                        'high_value_goals',
                        'high_value_subgoals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                        'start_obs',
                    ],
                )

        return batch

                    ],
                )

        return batch
