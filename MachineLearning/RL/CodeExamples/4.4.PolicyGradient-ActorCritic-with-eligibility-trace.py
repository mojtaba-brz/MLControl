import gymnasium as gym
import tensorflow as tf

from PyLib.ActorCritic import ActorNetwork, CriticNetwork, actor_network_default_dict, critic_network_default_dict
from PyLib.RenderTools import sim_gym_env

# Algorithm:
#   Loop forever:
#       s = env.reset
#       done = false
#       gamma_t = 1
#       z_w = z_theta = 0
#       While not done
#           a = actor_net(s)
#           s_next, r, done = env.step(a)
# 
#           delta   = r + gamma * critic_net(s_next) * (1-done) - critic_net(s)
#           z_theta = gamma * lambda_z_theta * z_theta + gamma_t * grad(ln(pi(a_t|s_t, theta)))
#           theta   = theta + alpha_t * delta * z_theta
#           z_w     = gamma * lambda_z_w * z_w + grad(v(s_t, w))
#           w       = w + alpha_v * delta * z_w
# 
#           gamma_t *= gamma

env_name = "CartPole-v1"
# env_name = "Acrobot-v1"
# env_name = "MountainCar-v0"
# env_name = "LunarLander-v3"

env = gym.make(env_name)

MAX_EPISODES = 100000
GAMMA = 0.99
SHOW_SIM_EVERY_N_EPISODE = 12000000
INITIAL_VALUE_NETWORK_LEARNING_RATE = 2e-2
INITIAL_ACTOR_LEARNING_RATE = INITIAL_VALUE_NETWORK_LEARNING_RATE * 1e-1
LEARNING_RATE_DECAY_COEF = 0.95 # An improvement mentioned by sutton in his RL book
LAMBDA_ACTOR = 0.1
LAMBDA_VALUE = 0.1
MAX_SIM_STEPS = 200

actor_params = actor_network_default_dict
actor_params['is_deterministic'] = False
actor_params['action_space_is_discrete'] = True
actor_params['soft_max_policy_temperature'] = 1.0
actor_params['hidden_layer_sizes'] = (256, )
actor_params['state_shape'] = env.observation_space.shape[0]
actor_params['action_shape'] = env.action_space.n
actor_params['optimizer'] = tf.keras.optimizers.Adam(learning_rate=INITIAL_ACTOR_LEARNING_RATE)
actor_params['ppo_epsilon'] = 0.05
actor_params['value_gamma'] = GAMMA

actor_net     = ActorNetwork(actor_params)

value_net_params = critic_network_default_dict
value_net_params['hidden_layer_sizes'] = (256, )
value_net_params['state_shape'] = env.observation_space.shape[0]
value_net = CriticNetwork(value_net_params)
value_net.optimizer._learning_rate = INITIAL_VALUE_NETWORK_LEARNING_RATE

actor_loss = None
ep_reward_filt = 0
actor_loss_filt = 0
value_loss_filt = 0
for ep in range(MAX_EPISODES):
    state, info = env.reset()
    done = False
    ep_reward = 0
    gamma_t = 1
    reward_array        = []
    action_array        = []
    state_array         = []
    next_state_array    = []
    z_actor = None
    z_value = None    
    num_steps = 0 
    while not done:
        action = actor_net.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated or num_steps >= MAX_SIM_STEPS
        ep_reward += reward

        reward_array.append(reward)
        action_array.append(action)
        state_array.append(state)
        next_state_array.append(next_state)
    
        reward_array_tf = tf.convert_to_tensor(reward_array, dtype=tf.float32)
        action_array_tf = tf.convert_to_tensor(action_array, dtype=tf.float32)
        state_array_tf = tf.convert_to_tensor(state_array, dtype=tf.float32)
        next_state_array_tf = tf.convert_to_tensor(next_state_array, dtype=tf.float32)
        values = value_net(state_array_tf)
        next_values = value_net(next_state_array_tf) * (1 - done)
        delta_array_tf = reward_array_tf + GAMMA*next_values - values
        with tf.GradientTape() as tape:
            actor_loss = actor_net.calc_pg_loss(gamma_t * delta_array_tf, state_array_tf, action_array_tf)
        actor_grads = tape.gradient(actor_loss, actor_net.trainable_variables)
        if z_actor is None:
            z_actor = actor_grads
        else:
            for i in range(len(actor_grads)):
                z_actor[i] = GAMMA * LAMBDA_ACTOR * z_actor[i] + actor_grads[i]
        actor_net.update(z_actor)
        
        with tf.GradientTape() as tape:
            values = value_net(state_array_tf)
            value_loss = tf.reduce_mean(tf.square(reward_array_tf + GAMMA*next_values - values))
        value_grads = tape.gradient(value_loss, value_net.trainable_variables)
        if z_value is None:
            z_value = value_grads
        else:
            for i in range(len(value_grads)):
                z_value[i] = GAMMA * LAMBDA_VALUE * z_value[i] + value_grads[i]
        value_net.update(z_value)

        gamma_t *= GAMMA
        state = next_state
        reward_array = []
        action_array = []
        state_array = []
        next_state_array = []
        actor_loss_filt += 0.05*(actor_loss - actor_loss_filt)
        value_loss_filt += 0.05*(value_loss - value_loss_filt)
        num_steps += 1

    if ep>0 and ep%SHOW_SIM_EVERY_N_EPISODE==0:
        sim_gym_env(actor_net, env_name, num_timesteps=MAX_SIM_STEPS)

    ep_reward_filt += 0.1*(ep_reward - ep_reward_filt)
    if actor_loss is not None and ep%10==0:
        print(f"Episode: {ep})\tReward: {ep_reward_filt:5.0f}, {ep_reward:5.0f}\tActor Loss: {actor_loss_filt:.5f}\tValue Loss: {value_loss_filt:.5f}\tActor LR: {actor_net.optimizer._learning_rate.numpy():.10f}")
        actor_net.optimizer._learning_rate *= LEARNING_RATE_DECAY_COEF
        value_net.optimizer._learning_rate *= LEARNING_RATE_DECAY_COEF

input()
sim_gym_env(actor_net, env_name, num_timesteps=MAX_SIM_STEPS)