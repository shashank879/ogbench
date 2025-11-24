from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize, RunningMeanStd
from utils.state_buffer import GoalBuffer


class DHPBufferAgent(flax.struct.PyTreeNode):
    """Discrete Hierarhical Planning (DHP) agent."""

    rng: Any
    network: Any
    goal_buffer: GoalBuffer = nonpytree_field()
    low_actor_val_norm: RunningMeanStd
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def merge_op(self, left, right):
        if self.config['merge_type'] == 'min':
            return jnp.minimum(left, right)
        elif self.config['merge_type'] == 'prod':
            return left * right
        else:
            raise NotImplementedError(self.config.merge_type)

    def low_value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_low_value')(batch['next_observations'], batch['low_value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['low_rewards'] + self.config['discount'] * batch['low_masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_low_value')(batch['observations'], batch['low_value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['low_rewards'] + self.config['discount'] * batch['low_masks'] * next_v1_t
        q2 = batch['low_rewards'] + self.config['discount'] * batch['low_masks'] * next_v2_t
        (v1, v2) = self.network.select('low_value')(batch['observations'], batch['low_value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def high_value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (left_next_v1_t, left_next_v2_t) = self.network.select('target_high_value')(batch['observations'], batch['high_value_subgoals'])
        (right_next_v1_t, right_next_v2_t) = self.network.select('target_high_value')(batch['high_value_subgoals'], batch['high_value_goals'])
        left_next_v_t = jnp.minimum(left_next_v1_t, left_next_v2_t)
        right_next_v_t = jnp.minimum(right_next_v1_t, right_next_v2_t)
        q = self.merge_op(
            batch['rewards_left'] + self.config['high_discount'] * batch['masks_left'] * left_next_v_t,
            batch['rewards_right'] + self.config['high_discount'] * batch['masks_right'] * right_next_v_t)

        (v1_t, v2_t) = self.network.select('target_high_value')(batch['observations'], batch['high_value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = self.merge_op(
            batch['rewards_left'] + self.config['high_discount'] * batch['masks_left'] * left_next_v1_t,
            batch['rewards_right'] + self.config['high_discount'] * batch['masks_right'] * right_next_v1_t)
        q2 = self.merge_op(
            batch['rewards_left'] + self.config['high_discount'] * batch['masks_left'] * left_next_v2_t,
            batch['rewards_right'] + self.config['high_discount'] * batch['masks_right'] * right_next_v2_t)
        (v1, v2) = self.network.select('high_value')(batch['observations'], batch['high_value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss."""
        v1, v2 = self.network.select('low_value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('low_value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()
        norm_v = self.low_actor_val_norm.normalize(v)

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'v': v.mean(),
            'v_std': v.std(),
            'next_v': nv.mean(),
            'norm_v_mean': norm_v.mean(),
            'norm_v_std': norm_v.std(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss."""
        v1, v2 = self.network.select(self.config['high_act_val_fn'])(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select(self.config['high_act_val_fn'])(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)
        )
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
            'v': v.mean(),
            'v_std': v.std(),
            'next_v': nv.mean(),
        }

    def high_actor_hierplan_loss(self, batch, grad_params):
        """Compute the high-level actor loss."""
        v = self.network.select(self.config['high_act_val_fn'])(batch['observations'], batch['high_actor_goals']).mean(0)
        left_nv = self.network.select(self.config['high_act_val_fn'])(batch['observations'], batch['high_actor_targets']).mean(0)
        right_nv = self.network.select(self.config['high_act_val_fn'])(batch['high_actor_targets'], batch['high_actor_goals']).mean(0)
        nv = self.merge_op(left_nv, right_nv)
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)
        )
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
            'v': v.mean(),
            'v_std': v.std(),
            'next_v': nv.mean(),
        }

    def decoder_loss(self, batch, grad_params):
        """Compute reconstruction loss for goal decoder.

        Decoder learns to invert the goal representation:
        goal_rep + emb_s -> emb_g
        """
        if grad_params:
            grad_params = {'modules_goal_decoder': grad_params['modules_goal_decoder']}
        # Get state encoder from value function
        if self.config['encoder'] is not None:
            # Extract state embeddings using value function's encoder
            emb_s = self.network.select('state_encoder')(batch['observations'])
            emb_g_target = self.network.select('state_encoder')(batch['low_value_goals'])
        else:
            # State-based: use observations directly
            emb_s = batch['observations']
            emb_g_target = batch['low_value_goals']

        # Compute goal representations
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_value_goals']], axis=-1)
        )

        # Decode: [goal_rep; emb_s] -> emb_g
        decoder_input = jax.lax.stop_gradient(jnp.concatenate([goal_reps, emb_s], axis=-1))
        emb_g_decoded = self.network.select('goal_decoder')(decoder_input, params=grad_params)

        # Reconstruction loss in embedding space
        if self.config['decoder_inv_loss']:
            target_goal_reps = jax.lax.stop_gradient(goal_reps)
            dec_goal_reps = self.network.select('goal_rep')(
                jnp.concatenate([batch['observations'], emb_g_decoded], axis=-1)
            )
            reconstruction_loss = jnp.mean((target_goal_reps - dec_goal_reps) ** 2)
            cosine_sim = jnp.sum(dec_goal_reps * target_goal_reps, axis=-1) / (
                jnp.linalg.norm(dec_goal_reps, axis=-1) * jnp.linalg.norm(target_goal_reps, axis=-1) + 1e-8
            )
        else:
            emb_g_target = jax.lax.stop_gradient(emb_g_target)
            reconstruction_loss = jnp.mean((emb_g_decoded - emb_g_target) ** 2)
            cosine_sim = jnp.sum(emb_g_decoded * emb_g_target, axis=-1) / (
                jnp.linalg.norm(emb_g_decoded, axis=-1) * jnp.linalg.norm(emb_g_target, axis=-1) + 1e-8
            )

        # Optional: cosine similarity loss for better direction matching
        cosine_loss = jnp.mean(1 - cosine_sim)

        total_decoder_loss = reconstruction_loss + 0.1 * cosine_loss

        return total_decoder_loss, {
            'decoder_loss': total_decoder_loss,
            'recon_mse': reconstruction_loss,
            'cosine_loss': cosine_loss,
            'cosine_sim': cosine_sim.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        low_value_loss, low_value_info = self.low_value_loss(batch, grad_params)
        for k, v in low_value_info.items():
            info[f'low_value/{k}'] = v

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
        for k, v in high_value_info.items():
            info[f'high_value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        if self.config['hierarchical_planner']:
            high_actor_loss, high_actor_info = self.high_actor_hierplan_loss(batch, grad_params)
        else:
            high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = low_value_loss + high_value_loss + low_actor_loss + high_actor_loss

        # Add decoder loss if enabled
        if self.config['use_goal_decoder'] and self.config['decoder_weight']:
            decoder_loss, decoder_info = self.decoder_loss(batch, grad_params)
            for k, v in decoder_info.items():
                info[f'decoder/{k}'] = v

            loss = loss + self.config['decoder_weight'] * decoder_loss

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def norm_update(self, batch):
        v = self.network.select('low_value')(batch['observations'], batch['low_actor_goals']).mean(0)
        new_low_actor_val_norm = self.low_actor_val_norm.update(v)
        info = {
            'low_actor/norm_mean': new_low_actor_val_norm.mean,
            'low_actor/norm_std': jnp.sqrt(new_low_actor_val_norm.var),
            'low_actor/norm_count': new_low_actor_val_norm.count,
        }
        return new_low_actor_val_norm, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'low_value')
        self.target_update(new_network, 'high_value')

        new_norm_net, norm_info = self.norm_update(batch)
        info.update(norm_info)

        return self.replace(network=new_network, low_actor_val_norm=new_norm_net, rng=new_rng), info

    def non_jit_update(self, batch, step):
        """Non-JIT updates including goal buffer population.

        This should be called after the JIT-compiled update() method.

        Args:
            batch: Training batch (should be on CPU/numpy for buffer operations)

        Returns:
            Dictionary with update statistics
        """
        info = {}

        update = (not self.goal_buffer.is_full()) or (step % self.config['buffer_update_freq'] == 0)

        # Update goal buffer if enabled
        if update and (self.goal_buffer is not None and self.config['use_goal_decoder']):
            # Add to buffer
            added_count = self.goal_buffer.add_batch(
                batch['low_value_goals'],
                # state_encoder=self.network.select('state_encoder'),
                # goal_rep_fn=lambda x,y: self.network.select('goal_rep')(jnp.concatenate([x, y], -1)),
                value_fn=lambda x,y: self.network.select('low_value')(x, y).mean(0),
            )
            info['buffer/added'] = added_count

            info.update({'buffer/'+k:v for k,v in self.goal_buffer.get_diagnostics().items()})

        return info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor.

        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        if self.config['hierarchical_planner']:
            return self.sample_hierarchical_actions(observations, goals, seed, temperature)
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    def sample_hierarchical_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
        use_buffer=True,
    ):
        """Sample actions from the actor.

        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        info = {}
        high_seed, low_seed = jax.random.split(seed)

        # Get current state embedding from value function's encoder
        if self.config['encoder'] is not None:
            emb_s = self.network.select('state_encoder')(observations)
        else:
            emb_s = observations

        def sample_subgoal(carry, _):
            """Sample one subgoal level."""
            current_goal = carry['goal']

            high_dist = self.network.select('high_actor')(observations, current_goal, temperature=temperature)
            goal_reps = high_dist.sample(seed=high_seed)
            goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

            # Decode goal_rep to embedding space
            decoder_input = jnp.concatenate([goal_reps, emb_s], axis=-1)
            emb_g_decoded = self.network.select('goal_decoder')(decoder_input)

            # Retrieve nearest goal from buffer
            retrieved_goal = self.goal_buffer.retrieve_nearest_embedding(
                goal_emb_query=None,
                goal_rep_query=goal_reps,
                current_state=observations,
                goal_rep_fn=lambda s,g: self.network.select('goal_rep')(
                    jnp.concatenate([s, g], axis=-1),
                ),
            )

            next_carry = {
                'goal': retrieved_goal,
            }
            return next_carry, (retrieved_goal, emb_g_decoded)

        init_carry = {
            'goal': goals,
        }
        final_carry, (subgoals, dec_goal_emb) = jax.lax.scan(
            sample_subgoal,
            init_carry,
            jnp.arange(self.config['hierplan_depth'])
        )

        subgoals = jnp.concatenate([goals[None], subgoals], axis=0)  # Shape: (depth+1, obs_dim)
        dec_goal_emb = jnp.concatenate([goals[None], dec_goal_emb], axis=0)  # Shape: (depth+1, obs_dim)
        low_value_pred = self.network.select('low_value')(jnp.stack([observations] * subgoals.shape[0], 0), subgoals).mean(0)  # Shape: (depth+1,)
        reachable = self.low_actor_val_norm.normalize(low_value_pred) >= self.config['reachable_thresh_val']
        cont = jax.lax.cumprod(1 - reachable, axis=0)  # Shape: (depth+1,)

        # Mark last subgoal as always reachable (fallback)
        reach_fb = jnp.concatenate([cont[:-1], jnp.zeros_like(cont[:1])], axis=0)
        reach_p1 = jnp.concatenate([jnp.ones_like(cont[:1]), cont[:-1]], axis=0)

        # Find first reachable subgoal
        first_reach = (reach_p1 - reach_fb).astype(jnp.float32)  # Shape: (depth+1,)
        first_reach_index = jnp.argmax(first_reach, axis=0)  # Scalar

        # Extract first reachable subgoal using weighted sum
        first_subgoal = jnp.sum(jnp.expand_dims(first_reach, [i+1 for i in range(len(subgoals.shape) - len(first_reach.shape))]) * subgoals, axis=0)

        info['subgoals'] = subgoals
        info['decoded_subgoals'] = dec_goal_emb
        info['subgoal_first_reach_index'] = first_reach_index

        low_dist = self.network.select('low_actor')(observations, first_subgoal, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions, info

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector.
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        # Define goal decoder: [goal_rep; emb_s] -> emb_g
        if config.get('use_goal_decoder', False):
            if config['encoder'] is not None:
                # For pixel-based: need to determine encoder output dimension
                encoder_output_dim = goal_rep_seq[0].mlp_hidden_dims[-1]
            else:
                # For state-based: embedding dimension = observation dimension
                # encoder_output_dim = ex_observations.shape[-1]
                encoder_output_dim = ex_observations.shape[-1]

            decoder_input_dim = config['rep_dim'] + encoder_output_dim
            goal_decoder_def = MLP(
                hidden_dims=(*config['value_hidden_dims'], encoder_output_dim),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        else:
            goal_decoder_def = None
            decoder_input_dim = None

        # Define the encoders that handle the inputs to the value and actor networks.
        # The subgoal representation phi([s; g]) is trained by the parameterized value function V(s, phi([s; g])).
        # The high-level actor predicts the subgoal representation phi([s; w]) for subgoal w given s and g.
        # The low-level actor predicts actions given the current state s and the subgoal representation phi([s; w]).
        if config['encoder'] is not None:
            # Pixel-based environments require visual encoders for state inputs, in addition to the pre-defined shared
            # encoder for subgoal representations.

            # Low Value: V^l(encoder^Vl(s), phi([s; g]))
            low_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_low_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # High Value: V^h(encoder^Vh([s; g]))
            high_value_encoder_def = GCEncoder(concat_encoder=encoder_module())
            target_high_value_encoder_def = GCEncoder(concat_encoder=encoder_module())
            # Low-level actor: pi^l(. | encoder^l(s), phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | encoder^h([s; g]))
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            # State-based environments only use the pre-defined shared encoder for subgoal representations.

            # Low Value: V^l(s, phi([s; g]))
            low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # High Value: V^h([s; g])
            high_value_encoder_def = None
            target_high_value_encoder_def = None
            # Low-level actor: pi^l(. | s, phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | s, g) (i.e., no encoder)
            high_actor_encoder_def = None

        # Define value and actor networks.
        low_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=low_value_encoder_def,
        )
        target_low_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_low_value_encoder_def,
        )

        high_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=high_value_encoder_def,
        )
        target_high_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_high_value_encoder_def,
        )

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=low_actor_encoder_def,
            )

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            low_value=(low_value_def, (ex_observations, ex_goals)),
            target_low_value=(target_low_value_def, (ex_observations, ex_goals)),
            high_value=(high_value_def, (ex_observations, ex_goals)),
            target_high_value=(target_high_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )

        # Add decoder if enabled
        if config.get('use_goal_decoder', False):
            ex_dec_input = jnp.zeros((1, decoder_input_dim))
            network_info['goal_decoder'] = (goal_decoder_def, (ex_dec_input,))
            network_info['state_encoder'] = (low_value_encoder_def.state_encoder, (ex_dec_input,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_low_value'] = params['modules_low_value']
        params['modules_target_high_value'] = params['modules_high_value']

        # Initialize goal buffer if decoder is enabled
        goal_buffer = None
        if config.get('use_goal_decoder', False):
            goal_buffer = GoalBuffer(
                capacity=config['goal_buffer_capacity'],
                reference_state=ex_observations[0],
            )

        return cls(rng, network=network, goal_buffer=goal_buffer, low_actor_val_norm=RunningMeanStd(), config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='dhpbuff',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            high_discount=0.9,  # High Discount factor.
            discount=0.99,  # Low Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actors.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            hierarchical_planner=True,
            reachable_thresh_val=-2,
            hierplan_depth=8,
            high_act_val_fn='high_value',  # [high_value, low_value]

            # Goal buffer hyperparameters
            use_goal_decoder=True,  # Whether to use goal decoder and buffer
            decoder_weight=0.1,  # Weight for decoder loss
            decoder_inv_loss=False,
            goal_buffer_capacity=2048,  # Buffer size
            goal_buffer_diversity=0.5,  # Diversity threshold in embedding space
            use_value_retrieval=False,  # Whether to use value-weighted retrieval
            buffer_update_freq=5000,

            # Dataset hyperparameters.
            dataset_class='DHPDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            high_value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            high_value_p_trajgoal=0.7,  # Probability of using a future state in the same trajectory as the value goal.
            high_value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            high_value_normal_subg_sample=False,  # Whether to use geometric sampling for future value goals.
            # high_value_min_dist=1,
            merge_type='min',
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
