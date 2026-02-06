from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize


class DHPExplAgent(flax.struct.PyTreeNode):
    """Discrete Hierarhical Planning (DHP) agent."""

    rng: Any
    network: Any
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

    def seq_value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        next_v_ts = self.network.select('target_seq_value')(batch['next_observations'], batch['seq_value_goals'])
        next_v_t = next_v_ts.min(0)
        q_mean = batch['seq_rewards'] + self.config['discount'] * batch['seq_masks'] * next_v_t

        v_ts = self.network.select('target_seq_value')(batch['observations'], batch['seq_value_goals'])
        v_t = v_ts.mean(0)
        adv = q_mean - v_t

        qs = batch['seq_rewards'] + self.config['discount'] * batch['seq_masks'] * next_v_ts
        vs = self.network.select('seq_value')(batch['observations'], batch['seq_value_goals'], params=grad_params)
        v = vs.mean(0)

        value_losses = self.expectile_loss(adv[None], qs - vs, self.config['expectile']).sum(0)
        value_loss = value_losses.mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def hier_value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        left_next_v_ts = self.network.select('target_hier_value')(batch['observations'], batch['hier_value_subgoals'])
        right_next_v_ts = self.network.select('target_hier_value')(batch['hier_value_subgoals'], batch['hier_value_goals'])
        left_next_v_t = left_next_v_ts.min(0)
        right_next_v_t = right_next_v_ts.min(0)
        q_mean = self.merge_op(
            batch['rewards_left'] + self.config['high_discount'] * batch['masks_left'] * left_next_v_t,
            batch['rewards_right'] + self.config['high_discount'] * batch['masks_right'] * right_next_v_t)

        v_ts = self.network.select('target_hier_value')(batch['observations'], batch['hier_value_goals'])
        v_t = v_ts.mean(0)
        adv = q_mean - v_t

        left_qs = batch['rewards_left'] + self.config['high_discount'] * batch['masks_left'] * left_next_v_ts
        right_qs = batch['rewards_right'] + self.config['high_discount'] * batch['masks_right'] * right_next_v_ts
        qs = self.merge_op(left_qs, right_qs)

        vs = self.network.select('hier_value')(batch['observations'], batch['hier_value_goals'], params=grad_params)
        v = vs.mean(0)

        value_losses = self.expectile_loss(adv[None], qs - vs, self.config['expectile']).sum(0)
        value_loss = value_losses.mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss."""
        v = self.network.select('seq_value')(batch['observations'], batch['low_actor_goals']).mean(0)
        nv = self.network.select('seq_value')(batch['next_observations'], batch['low_actor_goals']).mean(0)
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

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'v': v.mean(),
            'v_std': v.std(),
            'next_v': nv.mean(),
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
        vs = self.network.select(self.config['high_act_val_fn'])(batch['start_obs'], batch['observations'])
        nvs = self.network.select(self.config['high_act_val_fn'])(batch['start_obs'], batch['high_actor_targets'])
        v = vs.mean(0)
        nv, nv_std = nvs.mean(0), nvs.std(0)
        # Tries to minimize the gc-value function
        adv = (v - nv) + (.1 * nv_std)

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], batch['start_obs'], params=grad_params)
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
        left_nv = self.network.select(self.config['high_act_val_fn'])(batch['observations'], batch['high_actor_targets'])
        right_nv = self.network.select(self.config['high_act_val_fn'])(batch['high_actor_targets'], batch['high_actor_goals'])
        nvs = self.merge_op(left_nv, right_nv)
        nv, nv_std = nvs.mean(0), nvs.std(0)
        # Tries to minimize the gc-value function
        adv = (v - nv) + (.1 * nv_std)

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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        seq_value_loss, seq_value_info = self.seq_value_loss(batch, grad_params)
        for k, v in seq_value_info.items():
            info[f'seq_value/{k}'] = v

        hier_value_loss, hier_value_info = self.hier_value_loss(batch, grad_params)
        for k, v in hier_value_info.items():
            info[f'hier_value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        if self.config['hierarchical_planner']:
            high_actor_loss, high_actor_info = self.high_actor_hierplan_loss(batch, grad_params)
        else:
            high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = seq_value_loss + hier_value_loss + low_actor_loss + high_actor_loss

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'seq_value')
        self.target_update(new_network, 'hier_value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        init_obs=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor.

        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, init_obs, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

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

        # Define the encoders that handle the inputs to the value and actor networks.
        # The subgoal representation phi([s; g]) is trained by the parameterized value function V(s, phi([s; g])).
        # The high-level actor predicts the subgoal representation phi([s; w]) for subgoal w given s and g.
        # The low-level actor predicts actions given the current state s and the subgoal representation phi([s; w]).
        if config['encoder'] is not None:
            # Pixel-based environments require visual encoders for state inputs, in addition to the pre-defined shared
            # encoder for subgoal representations.

            # Low Value: V^l(encoder^Vl(s), phi([s; g]))
            seq_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_seq_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # High Value: V^h(encoder^Vh([s; g]))
            hier_value_encoder_def = GCEncoder(concat_encoder=encoder_module())
            target_hier_value_encoder_def = GCEncoder(concat_encoder=encoder_module())
            # Low-level actor: pi^l(. | encoder^l(s), phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | encoder^h([s; g]))
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            # State-based environments only use the pre-defined shared encoder for subgoal representations.

            # Low Value: V^l(s, phi([s; g]))
            seq_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_seq_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # High Value: V^h([s; g])
            hier_value_encoder_def = None
            target_hier_value_encoder_def = None
            # Low-level actor: pi^l(. | s, phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | s, g) (i.e., no encoder)
            high_actor_encoder_def = None

        # Define value and actor networks.
        seq_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=seq_value_encoder_def,
            num_ensembles=config['value_num_ensembles'],
        )
        target_seq_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_seq_value_encoder_def,
            num_ensembles=config['value_num_ensembles'],
        )

        hier_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=hier_value_encoder_def,
            num_ensembles=config['value_num_ensembles'],
        )
        target_hier_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_hier_value_encoder_def,
            num_ensembles=config['value_num_ensembles'],
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
            seq_value=(seq_value_def, (ex_observations, ex_goals)),
            target_seq_value=(target_seq_value_def, (ex_observations, ex_goals)),
            hier_value=(hier_value_def, (ex_observations, ex_goals)),
            target_hier_value=(target_hier_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_seq_value'] = params['modules_seq_value']
        params['modules_target_hier_value'] = params['modules_hier_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='dhpexpl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            value_num_ensembles=6,
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
            hierarchical_planner=False,
            reachable_thresh_val=-1.5,
            hierplan_depth=8,
            high_act_val_fn='seq_value',  # [hier_value, seq_value]

            # Dataset hyperparameters.
            dataset_class='DHPExplDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            hier_value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            hier_value_p_trajgoal=0.7,  # Probability of using a future state in the same trajectory as the value goal.
            hier_value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            hier_value_normal_subg_sample=False,  # Whether to use geometric sampling for future value goals.
            # hier_value_min_dist=1,
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
