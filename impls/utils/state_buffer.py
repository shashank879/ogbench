import jax
import jax.numpy as jnp
from typing import List, Optional


class GoalBuffer:
    """Buffer for storing diverse goal observations with their embeddings."""

    def __init__(self, capacity=1024, reference_state=None):
        self.capacity = capacity
        self.reference_state = reference_state  # Fixed reference
        self.goal_embeddings = []  # List of goal embeddings (emb_g)
        self.goal_observations = []  # List of actual goal observations

    def is_full(self):
        return len(self.goal_observations) >= self.capacity

    def add_batch(self, goal_observations, state_encoder=None, goal_rep_fn=None, value_fn=None):
        """Add a batch of goals efficiently using vectorized diversity selection.

        Args:
            goal_observations: Array of shape (batch_size, obs_dim)
            goal_embeddings: Array of shape (batch_size, emb_dim)

        Returns:
            Number of goals actually added
        """
        assert (state_encoder is None) or (goal_rep_fn is None), 'Only one can be specified'
        if state_encoder:
            self.reencode_all(state_encoder=state_encoder)
            goal_embeddings = jax.vmap(state_encoder)(goal_observations)
        elif goal_rep_fn:
            self.reencode_all(goal_rep_fn=goal_rep_fn)
            if self.reference_state is None:
                # Fallback: use zeros as reference
                self.reference_state = jnp.zeros_like(goal_observations[0])

            # Create batched input: [s_ref; g] for all g
            ref_repeated = jnp.broadcast_to(self.reference_state[None], (len(goal_observations), *self.reference_state.shape))

            # Compute embeddings in batch
            goal_embeddings = jax.vmap(goal_rep_fn)(ref_repeated, goal_observations)
        else:
            goal_embeddings = jnp.zeros((len(goal_observations), 1))

        initial_size = len(self.goal_embeddings)

        # Combine new batch with existing buffer
        if len(self.goal_embeddings) == 0:
            combined_embeddings = goal_embeddings
            combined_observations = goal_observations
        else:
            combined_embeddings = jnp.concatenate([
                jnp.stack(self.goal_embeddings), 
                goal_embeddings
            ], axis=0)
            combined_observations = jnp.concatenate([
                jnp.stack(self.goal_observations),
                goal_observations
            ], axis=0)

        # If under capacity, just add everything
        if len(combined_embeddings) <= self.capacity:
            self.goal_embeddings = [combined_embeddings[i] for i in range(len(combined_embeddings))]
            self.goal_observations = [combined_observations[i] for i in range(len(combined_observations))]
            return len(goal_observations)

        # Need to prune - select top-K most diverse using farthest point sampling
        if value_fn:
            selected_indices = self._select_diverse_subset(
                combined_observations, 
                self.capacity,
                lambda x,y: -value_fn(x,y),
            )
        else:
            selected_indices = self._select_diverse_subset(
                combined_embeddings, 
                self.capacity
            )

        # Update buffer with selected goals
        self.goal_embeddings = [combined_embeddings[i] for i in selected_indices]
        self.goal_observations = [combined_observations[i] for i in selected_indices]

        # Count how many new goals were kept
        new_goals_kept = jnp.sum(selected_indices >= initial_size)
        return int(new_goals_kept)

    def _select_diverse_subset(self, embeddings, k, value_fn=None):
        """Select k most diverse embeddings using farthest point sampling.

        This is a greedy algorithm that iteratively selects the point farthest
        from already selected points, maximizing minimum pairwise distances.

        Args:
            embeddings: Array of shape (n, emb_dim)
            k: Number of points to select

        Returns:
            Array of k selected indices
        """
        n = len(embeddings)
        if k >= n:
            return jnp.arange(n)

        # Compute all pairwise distances once (vectorized)
        # Shape: (n, n)
        if value_fn:
            pairwise_distances = self._compute_pairwise_chunked(
                embeddings, value_fn)
        else:
            x = jnp.tile(embeddings[:, None, :], (1, n, 1))
            y = jnp.tile(embeddings[None, :, :], (n, 1, 1))
            distance_fn = lambda x,y: jnp.linalg.norm(x - y, axis=-1)
            pairwise_distances = distance_fn(x, y)

        # Initialize with random point (or first point for determinism)
        selected = jnp.zeros(k, dtype=jnp.int32)
        selected = selected.at[0].set(0)

        # Track minimum distance from each point to selected set
        min_distances = pairwise_distances[0]

        # Greedily select k-1 more points
        for i in range(1, k):
            # Select point with maximum distance to selected set
            next_idx = jnp.argmax(min_distances)
            selected = selected.at[i].set(next_idx)

            # Update minimum distances
            new_distances = pairwise_distances[next_idx]
            min_distances = jnp.minimum(min_distances, new_distances)

        return selected

    def _compute_pairwise_chunked(self, observations, value_fn, chunk_size=128):
        """Compute pairwise value-based distances in chunks to avoid OOM.

        Args:
            observations: Array of shape (n, obs_dim)
            value_fn: Function that takes (s, g) and returns negative value
            chunk_size: Number of pairs to process at once

        Returns:
            Pairwise distances array of shape (n, n)
        """
        n = len(observations)
        pairwise_distances = jnp.zeros((n, n))

        # Process in chunks of rows
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            chunk_obs_i = observations[i:end_i]

            # For each chunk of i, compute distances to all j
            chunk_distances = []
            for j in range(0, n, chunk_size):
                end_j = min(j + chunk_size, n)
                chunk_obs_j = observations[j:end_j]

                # Compute (chunk_i_size, chunk_j_size) distance matrix
                # Expand dimensions for broadcasting
                s_expanded = jnp.broadcast_to(chunk_obs_i[:, None, ...], (len(chunk_obs_i), len(chunk_obs_j), *chunk_obs_i[0].shape))
                g_expanded = jnp.broadcast_to(chunk_obs_j[None, :, ...], (len(chunk_obs_i), len(chunk_obs_j), *chunk_obs_j[0].shape))

                # value_fn should be vectorized to handle batched inputs
                chunk_dist = value_fn(s_expanded, g_expanded)
                chunk_distances.append(chunk_dist)

            # Concatenate chunks for this row range
            row_distances = jnp.concatenate(chunk_distances, axis=1)
            pairwise_distances = pairwise_distances.at[i:end_i, :].set(row_distances)

        return pairwise_distances

    def retrieve_nearest_embedding(self, goal_emb_query=None, goal_rep_query=None, current_state=None, goal_rep_fn=None):
        """Retrieve observation nearest to query embedding.

        Args:
            emb_g_query: Query embedding in goal embedding space
            current_state: Current state observation (for value-based retrieval)
            network: HIQL network (for value function queries)
            use_value: Whether to use value-weighted retrieval

        Returns:
            Retrieved goal observation (as JAX array)
        """

        observations_array = jnp.stack(self.goal_observations)

        if goal_rep_fn:
            # Value-weighted retrieval: balance distance and reachability
            current_state_expanded = jnp.broadcast_to(current_state[None], (len(self.goal_observations), *current_state.shape))
            pred_goal_rep = jax.vmap(goal_rep_fn)(current_state_expanded, observations_array)
            distances = jnp.linalg.norm(goal_rep_query - pred_goal_rep, axis=1)
            best_idx = distances.argmin()
        else:
            embeddings_array = jnp.stack(self.goal_embeddings)
            # Compute distances in embedding space (vectorized)
            distances = jnp.linalg.norm(embeddings_array - goal_emb_query, axis=1)
            # Pure nearest neighbor in embedding space
            best_idx = distances.argmin()

        # Use JAX indexing to retrieve the observation (JIT-compatible)
        return observations_array[best_idx]

    def to_jax_arrays(self):
        """Convert buffer to JAX arrays for efficient operations."""
        if len(self.goal_embeddings) == 0:
            return None, None
        return jnp.stack(self.goal_embeddings), jnp.stack(self.goal_observations)

    def reencode_all(self, state_encoder=None, goal_rep_fn=None):
        """Re-encode all stored goals with updated encoder and re-select diverse subset.

        Args:
            state_encoder: Updated state encoder function that takes batched input

        Returns:
            Number of goals after re-encoding and diversity selection
        """
        if len(self.goal_observations) == 0:
            return 0

        # Re-encode all goals in batch (much faster than one-by-one)
        observations_array = jnp.stack(self.goal_observations)

        if goal_rep_fn:
            # Re-compute embeddings: phi([s_ref; g])
            if self.reference_state is None:
                self.reference_state = jnp.zeros_like(observations_array[0])

            ref_repeated = jnp.broadcast_to(self.reference_state[None], (len(self.goal_observations), *self.reference_state.shape))
            new_embeddings = jax.vmap(goal_rep_fn)(ref_repeated, observations_array)
        else:
            new_embeddings = state_encoder(observations_array)

        # Re-run diversity selection with new embeddings
        if len(new_embeddings) > self.capacity:
            selected_indices = self._select_diverse_subset(new_embeddings, self.capacity)
            self.goal_embeddings = [new_embeddings[i] for i in selected_indices]
            self.goal_observations = [observations_array[i] for i in selected_indices]
        else:
            self.goal_embeddings = [new_embeddings[i] for i in range(len(new_embeddings))]
            # Keep observations as-is

        return len(self.goal_embeddings)

    def get_diagnostics(self):
        """Get diagnostic metrics about buffer state."""
        if len(self.goal_embeddings) == 0:
            return {}

        embeddings_array = jnp.stack(self.goal_embeddings)

        # Pairwise distances
        diff = embeddings_array[:, None, :] - embeddings_array[None, :, :]
        pairwise_distances = jnp.linalg.norm(diff, axis=-1)

        # Mask diagonal
        mask = 1 - jnp.eye(len(embeddings_array))
        masked_distances = pairwise_distances + (1 - mask) * 1e10

        # Nearest neighbor distances
        nn_distances = masked_distances.min(axis=1)

        # Centroid and coverage
        centroid = embeddings_array.mean(axis=0)
        radii = jnp.linalg.norm(embeddings_array - centroid, axis=1)

        return {
            'size': len(self.goal_embeddings),
            'utilization': len(self.goal_embeddings) / self.capacity,
            'min_pairwise_distance': float(masked_distances.min()),
            'mean_nn_distance': float(nn_distances.mean()),
            'std_nn_distance': float(nn_distances.std()),
            'max_nn_distance': float(nn_distances.max()),
            'coverage_radius': float(radii.max()),
            'mean_coverage_radius': float(radii.mean()),
        }
