import tensorflow as tf
import numpy as np
from keras.layers import Dense
import tensorflow_probability as tfp
import gymnasium as gym

from .BasicPolicyRelatedTools import SoftMaxPolicy

actor_network_default_dict = { 'hidden_layer_sizes':(200,), 
                               'is_deterministic':False,
                               'action_space_is_discrete':True,
                               'min_stdev':1e-6, # For gaussian polices (continuous)
                               'mean_layer_activation_function':'tanh',
                               'hidden_layers_activation_function':'relu',
                               'action_shape':1,
                               'state_shape':1,
                               'action_low_values':-1.,
                               'action_high_values':1.,
                               'action_max':1.0,
                               'optimizer':tf.keras.optimizers.Adam(learning_rate=0.001),
                               'adv_gamma': 1,
                               'value_gamma':0.99,
                               'ppo_epsilon':0.1}

critic_network_default_dict = { 'hidden_layer_sizes':(100,),
                               'output_layer_activation_function':'linear',
                               'hidden_layers_activation_function':'relu',
                               'state_shape':1,
                               'mode':'V',
                               'optimizer':tf.keras.optimizers.Adam(learning_rate=0.001),}

class ActorNetwork(tf.keras.Model):
    # It's also called policy network
    g_actor_count = 1
    def __init__(self, 
                 actor_params:dict = actor_network_default_dict,
                 name=f"Actor{g_actor_count}"):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.action_space_is_discrete = actor_params['action_space_is_discrete']

        if self.action_space_is_discrete:
            self.soft_max_policy = SoftMaxPolicy(temperature=1.0)

        self.action_shape                       = actor_params['action_shape']
        self.state_shape                        = actor_params['state_shape']
        self.action_bound                       = (actor_params['action_low_values'], actor_params['action_high_values'])
        self.hidden_layer_sizes                 = actor_params['hidden_layer_sizes']
        self.is_deterministic                   = actor_params['is_deterministic']
        self.min_stdev                          = actor_params['min_stdev']
        self.action_max                         = actor_params['action_max']
        self.hidden_layers_activation_function  = actor_params['hidden_layers_activation_function']
        self.mean_layer_activation_function     = actor_params['mean_layer_activation_function']
        self.optimizer                          = actor_params['optimizer']
        self.adv_gamma                          = actor_params['adv_gamma']
        self.value_gamma                        = actor_params['value_gamma']
        self.ppo_epsilon                        = actor_params['ppo_epsilon']
        
        self.hidden_layers = []
        # Create hidden layers
        for h in self.hidden_layer_sizes:
            self.hidden_layers.append(Dense(h, activation=self.hidden_layers_activation_function))

        self.mean = Dense(self.action_shape, activation=self.mean_layer_activation_function)
        if not self.is_deterministic:
            self.stdev = Dense(self.action_shape, activation='softplus')
        
        self.build_networks()

        self.g_actor_count += 1
        
    def call(self, state):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        mean = self.mean(x)
        
        if self.action_space_is_discrete:
            return mean # action preferences
        elif self.is_deterministic:
            return mean * self.action_max # action
        else:
            stdev = self.stdev(x)
            stdev = tf.clip_by_value(stdev, self.min_stdev, np.inf)
            return mean, stdev # action distribution

    def sample_normal(self, state, reparameterize=True):
        if self.action_space_is_discrete:
            raise RuntimeError("...")
        if self.is_deterministic:
            raise RuntimeError("Deterministic actor does not support sample_normal; use call() instead.")
        
        mean, stdev = self.call(state)

        if reparameterize:
            # During training: Reparameterization trick
            # sample epsilon from standard normal, and then transform it using mean and stdev
            epsilon = tf.random.normal(shape=mean.shape)
            actions = mean + epsilon * stdev
        else:
            # During action selection, directly sample from the normal distribution
            probabilities = tfp.distributions.Normal(mean, stdev)
            actions = probabilities.sample()

        # Apply tanh squashing
        action = tf.math.tanh(actions)

        # Calculate log probabilities (using the original normal distribution)
        probabilities = tfp.distributions.Normal(mean, stdev)
        log_probs = probabilities.log_prob(actions)

        # Apply the squashing correction
        log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + self.min_stdev)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        action = tf.clip_by_value(action * self.action_max, self.action_bound[0], self.action_bound[1])
        return action, log_probs
    
    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        if self.action_space_is_discrete:
            act_pref = self(state)
            action = self.soft_max_policy.get_action(act_pref.numpy()[0])
        elif self.is_deterministic:
            action = self(state)
            action = tf.clip_by_value(action, self.action_bound[0], self.action_bound[1])
        else:
            mean, stdv = self(state)
            action = tf.random.normal(shape=mean.shape, mean=mean, stddev=stdv)
            action = tf.clip_by_value(action * self.action_max, self.action_bound[0], self.action_bound[1])

        return action
    
    def build_networks(self):
        dummy_state = tf.zeros([1, self.state_shape])
        self(dummy_state)

    def get_log_prob(self, state):
        if self.action_space_is_discrete:
            act_pref = self(state)
            probs = self.soft_max_policy.get_probabilities_tf(act_pref)
        else:
            pass

        return tf.math.log(probs)

    def get_action_log_prob(self, states, actions):
        if self.action_space_is_discrete:
            act_pref = self(states)
            probs = self.soft_max_policy.get_action_probabilities_tf(act_pref, actions)
        else:
            pass

        return tf.math.log(probs)
    
    def update(self, actor_grads):
        self.optimizer.apply_gradients(zip(actor_grads, self.trainable_variables))

    def calc_pg_loss(self, q, states, actions):
        return -tf.reduce_mean(self.get_action_log_prob(states, actions) * q)

    def calc_pg_mc_loss(self, rewards, states, actions):
        gamma_t_G_t = []
        gamma_t = 1
        for i in range(len(rewards)):
            G_t = 0
            for reward in rewards[i:]:
                G_t = self.value_gamma * G_t + reward
            gamma_t_G_t.append(gamma_t * G_t)
            gamma_t *= self.value_gamma
            
        return -tf.reduce_mean(self.get_action_log_prob(states, actions) * gamma_t_G_t)

    def calc_pg_mc_with_baseline_loss(self, rewards, states, actions, baseline_values):
        gamma_t_delta_t = []
        gamma_t = 1
        for i in range(len(rewards)):
            G_t = 0
            for reward in rewards[i:]:
                G_t = self.value_gamma * G_t + reward
            gamma_t_delta_t.append(gamma_t * (G_t - baseline_values[i]))
            gamma_t *= self.value_gamma
            
        return -tf.reduce_mean(self.get_action_log_prob(states, actions) * gamma_t_delta_t)
    
    def calc_ppo_clip_loss(self, addvantage_array_tf, actor_net_old, state_array, action_arrray):
        ratio = tf.exp(self.get_action_log_prob(state_array, action_arrray) - tf.stop_gradient(actor_net_old.get_action_log_prob(state_array, action_arrray)))
        
        actor_loss = tf.minimum(ratio*addvantage_array_tf, 
                                tf.clip_by_value(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)*addvantage_array_tf)
        
        return -tf.reduce_mean(actor_loss)

class CriticNetwork(tf.keras.Model):
    # Q((x, u)) or Q((s, a))
    # V(x) or V(s)
    # mode can be either one of the following:
    #   - 'Q' not working
    #   - 'V'
    g_critic_count = 1
    def __init__(self, params:dict = critic_network_default_dict, name=f"Critic{g_critic_count}"):
        assert params['mode'] == 'V', "Undefined mode"

        super(CriticNetwork, self).__init__()
        self.name = name
        self.state_shape = params['state_shape']
        self.hidden_layer_sizes = params['hidden_layer_sizes']
        self.output_layer_activation_function = params['output_layer_activation_function']
        self.hidden_layers_activation_function = params['hidden_layers_activation_function']
        self.optimizer = params['optimizer']
        
        self.hidden_layers = []
        # Create hidden layers
        for h in self.hidden_layer_sizes:
            self.hidden_layers.append(Dense(h, activation=self.hidden_layers_activation_function))
        self.critic = Dense(1, activation=self.output_layer_activation_function)
            
        self.g_critic_count += 1
        self.build_networks()

    def call(self, state):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        critic = self.critic(x)
        return critic # tf.squeeze(critic, axis=1)

    def build_networks(self):
        dummy_state = tf.zeros([1, self.state_shape])
        self(dummy_state)

    def update(self, critic_grads):
        self.optimizer.apply_gradients(zip(critic_grads, self.trainable_variables))

class ActorCritic:
    def __init__(self, env, name, global_ac=None):
        self.name = name
        state_shape = env.observation_space.shape[0]
        self.action_bound = [env.action_space.low, env.action_space.high]
        self.beta = 0.01  # Entropy coefficient
        
        # Initialize networks by calling them once
        dummy_state = tf.zeros([1, state_shape])
        
        self.actor = ActorNetwork(env)
        self.actor(dummy_state)  # Build the model
        
        self.critic = CriticNetwork(env)
        self.critic(dummy_state)  # Build the model
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        
        if global_ac is not None:  # For worker networks
            self.global_actor = global_ac.actor
            self.global_critic = global_ac.critic
    
    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        mean, variance = self.actor(state)
        mean = mean * self.action_bound[1]
        variance = variance + 1e-4
        
        normal_dist = tf.random.normal(shape=mean.shape)
        action = mean + tf.sqrt(variance) * normal_dist
        action = tf.clip_by_value(action, self.action_bound[0], self.action_bound[1])
        
        return action[0]
    
    def update(self, states, actions, target_values):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            # Actor forward pass
            mean, variance = self.actor(states)
            mean = mean * self.action_bound[1]
            variance = variance + 1e-4
            
            # Create normal distribution
            dist = tfp.distributions.Normal(mean, tf.sqrt(variance))
            log_prob = tf.reduce_sum(dist.log_prob(actions), axis=1, keepdims=True)
            entropy = tf.reduce_sum(dist.entropy(), axis=1, keepdims=True)
            
            # Critic forward pass
            values = self.critic(states)
            td_error = target_values - values
            
            # Define losses
            actor_loss = -tf.reduce_mean(log_prob * tf.stop_gradient(td_error) + self.beta * entropy)
            critic_loss = tf.reduce_mean(tf.square(td_error))
        
        # Compute gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        
        # Clip gradients
        actor_grads = [tf.clip_by_norm(grad, 40) for grad in actor_grads if grad is not None]
        critic_grads = [tf.clip_by_norm(grad, 40) for grad in critic_grads if grad is not None]
        
        # Create gradient-variable pairs
        actor_grad_vars = list(zip(actor_grads, self.global_actor.trainable_variables))
        critic_grad_vars = list(zip(critic_grads, self.global_critic.trainable_variables))
        
        # Apply gradients if they exist
        if actor_grad_vars:
            self.actor_optimizer.apply_gradients(actor_grad_vars)
        if critic_grad_vars:
            self.critic_optimizer.apply_gradients(critic_grad_vars)
        
        # Update local network
        self.pull_from_global()
        
    def pull_from_global(self):
        for l, g in zip(self.actor.variables, self.global_actor.variables):
            l.assign(g)
        for l, g in zip(self.critic.variables, self.global_critic.variables):
            l.assign(g)
            
class GlobalAgent:
    def __init__(self, env, actor_hidden_sizes=(200,)):
        self.actor = ActorNetwork(env, actor_hidden_sizes)
        self.critic = CriticNetwork(env)
        
        self.actor.build_networks()
        self.critic.build_networks()
        
        self.actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    
    def update(self, all_gradients):
        """Perform synchronous update with accumulated gradients from all workers"""
        # Unzip gradients
        actor_gradients_list = [g[0] for g in all_gradients]
        critic_gradients_list = [g[1] for g in all_gradients]
        
        # Average gradients across all workers
        avg_actor_grads = [
            tf.reduce_mean([grads[i] for grads in actor_gradients_list], axis=0)
            for i in range(len(actor_gradients_list[0]))
        ]
        avg_critic_grads = [
            tf.reduce_mean([grads[i] for grads in critic_gradients_list], axis=0)
            for i in range(len(critic_gradients_list[0]))
        ]
        
        # Apply averaged gradients
        self.actor_optimizer.apply_gradients(
            zip(avg_actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(
            zip(avg_critic_grads, self.critic.trainable_variables))


