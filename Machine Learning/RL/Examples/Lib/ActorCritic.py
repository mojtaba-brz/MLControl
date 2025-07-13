import tensorflow as tf
import numpy as np
from keras import Model, layers
from keras.layers import Dense
import tensorflow_probability as tfp

class ActorNetwork(tf.keras.Model):
    # It's also called policy network
    g_actor_count = 1
    def __init__(self, env, 
                 hidden_layer_sizes = (200,), 
                 is_deterministic = False, 
                 min_stdev = 1e-6, # action noise
                 name=f"Actor{g_actor_count}",
                 mean_layer_activation_function = "tanh",
                 hidden_layers_activation_function = "relu"):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.action_shape = env.action_space.shape[0]
        self.state_shape = env.observation_space.shape[0]
        self.action_bound = [env.action_space.low, env.action_space.high]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.is_deterministic = is_deterministic
        self.min_stdev = min_stdev
        self.mean_layer_activation_function = mean_layer_activation_function
        
        if self.mean_layer_activation_function == "tanh":
            self.action_max = max(np.abs(env.action_space.high), np.abs(env.action_space.low)) # Used as a gain after tanh function of the output layer
        else:
            self.action_max = 1.
        
        self.hidden_layers = []
        # Create hidden layers
        for h in hidden_layer_sizes:
            self.hidden_layers.append(Dense(h, activation=hidden_layers_activation_function))

        self.mean = Dense(self.action_shape, activation=mean_layer_activation_function)
        if not is_deterministic:
            self.stdev = Dense(self.action_shape, activation='softplus')
        
        self.g_actor_count += 1
        
    def call(self, state):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        mean = self.mean(x)
        
        if self.is_deterministic:
            return mean * self.action_max # action
        else:
            stdev = self.stdev(x)
            stdev = tf.clip_by_value(stdev, self.min_stdev, np.inf)
            return mean, stdev # action distribution

    def sample_normal(self, state, reparameterize=True):
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
    
    def sample_action(self, state):
        mean, stdv = self(state)
        action = tf.random.normal(shape=mean.shape, mean=mean, stddev=stdv)
        action = tf.clip_by_value(action * self.action_max, self.action_bound[0], self.action_bound[1])
        return action
    
    def build_networks(self):
        dummy_state = tf.zeros([1, self.state_shape])
        self(dummy_state)
    
class CriticNetwork(tf.keras.Model):
    # Q((x, u)) or Q((s, a))
    # V(x) or V(s)
    g_critic_count = 1
    def __init__(self, env,
                 hidden_layer_sizes = (100,), 
                 name=f"Critic{g_critic_count}",
                 critic_layer_activation_function = "linear",
                 hidden_layers_activation_function = "relu"):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.critic_layer_activation_function = critic_layer_activation_function
        
        self.hidden_layers = []
        # Create hidden layers
        for h in hidden_layer_sizes:
            self.hidden_layers.append(Dense(h, activation=hidden_layers_activation_function))
        self.critic = Dense(1, activation=critic_layer_activation_function)

        self.g_critic_count += 1
        
    def call(self, state_action):
        x = state_action
        for layer in self.hidden_layers:
            x = layer(x)
        critic = self.critic(x)
        return tf.squeeze(critic, axis=1)

    def build_networks(self):
        dummy_state = tf.zeros([1, self.state_shape+self.action_shape])
        self(dummy_state)

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
    
    def select_action(self, state):
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
