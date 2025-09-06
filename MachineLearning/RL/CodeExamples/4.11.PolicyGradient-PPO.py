import gymnasium as gym
import tensorflow as tf

from PyLib.ActorCritic import ActorNetwork, CriticNetwork, actor_network_default_dict, critic_network_default_dict
from PyLib.RenderTools import sim_gym_env

# Definitions:
# Advantage:
#   delta_t = r_t + gamma * v(s_{t+1}) - v(s_t)
#   A_t = delta_t + gamma*lambda*delta_{t+1} + ... + (gamma*lambda)^{T-t+1} * delta(T-1)
#   for lambda = 1 -->
#       A_t = -v(s_t) + r_t + gamma*r_{t+1} + ... + (gamma)^{T-t+1} * v(s_T)
# 
# Algorithm:
#   Loop forever:
#       for each actor in actors:
#           Generate an episode
#           Calculate the advantage
#       for i in range(NUM_OF_EPOCHES):
#           r = pi(s, a)/pi_old(s, a)
#           actor_loss = min(r * A, clip(r, 1-eps, 1+eps)*A)
#           update actor using grad(actor_loss)
#       update actors networks
# 
# Ref: Schulman 2017, PPO Algorithm

env_name = "CartPole-v1"
# env_name = "LunarLander-v3"

env = gym.make(env_name)

MAX_EPISODES = 100000
GAMMA = 0.99
SHOW_SIM_EVERY_N_EPISODE = 120000
INITIAL_VALUE_NETWORK_LEARNING_RATE = 2e-4
INITIAL_ACTOR_LEARNING_RATE = 2e-3
LEARNING_RATE_DECAY_COEF = 0.9999 # An improvement mentioned by sutton in his RL book
MAX_SIM_STEPS = 600
PPO_EPSILON = 0.2
NUM_OF_EPOCHES_ACTOR = 13
NUM_OF_EPOCHES_VALUE = 3
NUM_OF_ACTORS  = 7
LAMBDA = 1
ALPHA_UPDATE_EVERY_N_EPISODES = 3

actor_params = actor_network_default_dict
actor_params['is_deterministic'] = False
actor_params['action_space_is_discrete'] = True
actor_params['soft_max_policy_temperature'] = 1.0
actor_params['hidden_layer_sizes'] = (256, )
actor_params['state_shape'] = env.observation_space.shape[0]
actor_params['action_shape'] = env.action_space.n
actor_params['optimizer'] = tf.keras.optimizers.Adam(learning_rate=INITIAL_ACTOR_LEARNING_RATE)
actor_params['ppo_epsilon'] = PPO_EPSILON
actor_params['value_gamma'] = GAMMA

actor_net     = ActorNetwork(actor_params)
actor_net_old = ActorNetwork(actor_params)
actor_net_old.set_weights(actor_net.get_weights())

value_net_params = critic_network_default_dict
value_net_params['hidden_layer_sizes'] = (512, )
value_net_params['state_shape'] = env.observation_space.shape[0]
value_net_params['optimizer'] = tf.keras.optimizers.Adam(learning_rate=INITIAL_VALUE_NETWORK_LEARNING_RATE)
value_net = CriticNetwork(value_net_params)

def generate_one_episode(env, actor_net):
    state, _ = env.reset()
    done = False
    ep_reward = 0
    reward_array        = []
    action_array        = []
    state_array         = []
    next_state_array    = []
    num_steps = 0 
    while not done:
        action = actor_net.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated or num_steps >= MAX_SIM_STEPS
        ep_reward += reward

        reward_array.append(reward)
        action_array.append(action)
        state_array.append(state)
        next_state_array.append(next_state)

        state = next_state
        num_steps += 1
    
    state_array_tf = tf.convert_to_tensor(state_array, dtype=tf.float32)
    action_array_tf = tf.convert_to_tensor(action_array, dtype=tf.float32)
    reward_array_tf = tf.convert_to_tensor(reward_array, dtype=tf.float32)
    next_state_array_tf = tf.convert_to_tensor(next_state_array, dtype=tf.float32)

    return state_array_tf, action_array_tf, reward_array_tf, next_state_array_tf

def calc_episode_advatages(state_array, reward_array, next_state_array):
    n = len(reward_array)
    reward_array = tf.reshape(reward_array, (-1, 1))
    delta = tf.stop_gradient(reward_array + GAMMA * value_net(next_state_array) - value_net(state_array))
    addvantage_array = [delta[-1],]
    for i in range(n-2, -1, -1):
        addvantage_array.append(delta[i] + (LAMBDA * GAMMA) * addvantage_array[-1])
    addvantage_array.reverse()
    addvantage_array_tf = tf.convert_to_tensor(addvantage_array, dtype=tf.float32)

    return addvantage_array_tf

actor_loss = None
ep_reward_filt = 0
actor_loss_filt = 0
value_loss_filt = 0
for ep in range(1, MAX_EPISODES+1):
    state_array_list = []
    action_array_list = []
    reward_array_list = []
    next_state_array_list = []
    addvantage_array_list = []
    ep_reward = []
    for _ in range(NUM_OF_ACTORS):
        state_array_tf, action_array_tf, reward_array_tf, next_state_array_tf = generate_one_episode(env, actor_net)
        addvantage_array_tf = calc_episode_advatages(state_array_tf, reward_array_tf, next_state_array_tf)
        state_array_list.append(state_array_tf)
        action_array_list.append(action_array_tf)
        reward_array_list.append(reward_array_tf)
        next_state_array_list.append(next_state_array_tf)
        addvantage_array_list.append(addvantage_array_tf)
        ep_reward.append(tf.reduce_sum(reward_array_tf))

    ep_reward = tf.reduce_mean(ep_reward)
    state_array_list_tf      = tf.concat(state_array_list, 0)
    action_array_list_tf     = tf.concat(action_array_list, 0)
    reward_array_list_tf     = tf.reshape(tf.concat(reward_array_list, 0), (-1, 1))
    next_state_array_list_tf = tf.concat(next_state_array_list, 0)
    addvantage_array_list_tf = tf.concat(addvantage_array_list, 0)

    actor_net_weights_old = actor_net.get_weights()
    for _ in range(NUM_OF_EPOCHES_ACTOR):
        with tf.GradientTape() as tape:
            actor_loss = actor_net.calc_ppo_clip_loss(addvantage_array_list_tf, actor_net_old, state_array_list_tf, action_array_list_tf)
        actor_grads = tape.gradient(actor_loss, actor_net.trainable_variables)
        actor_net.update(actor_grads)
        actor_loss_filt += 0.05*(actor_loss - actor_loss_filt)
    actor_net_old.set_weights(actor_net_weights_old)

    next_values = value_net(next_state_array_list_tf)
    for _ in range(NUM_OF_EPOCHES_VALUE):
        with tf.GradientTape() as tape:
            values = value_net(state_array_list_tf)
            value_loss = tf.reduce_mean(tf.square(reward_array_list_tf + GAMMA*next_values - values))
        value_grads = tape.gradient(value_loss, value_net.trainable_variables)
        value_net.update(value_grads)
        value_loss_filt += 0.05*(value_loss - value_loss_filt)

    if ep%SHOW_SIM_EVERY_N_EPISODE==0:
        sim_gym_env(actor_net, env_name, num_timesteps=MAX_SIM_STEPS)

    ep_reward_filt += 0.1*(ep_reward - ep_reward_filt)
    print(f"Episode: {ep})\t"
          f"Reward: {ep_reward_filt:5.0f}, {ep_reward:5.0f}\t"
          f"Actor Loss: {actor_loss_filt:.5f}\t"
          f"Value Loss: {value_loss_filt:.5f}, {value_loss:.5f}\t"
          f"Actor LR: {actor_net.optimizer._learning_rate.numpy():.10f}")
    if ep%ALPHA_UPDATE_EVERY_N_EPISODES==0:
        actor_net.optimizer._learning_rate *= LEARNING_RATE_DECAY_COEF
        value_net.optimizer._learning_rate *= LEARNING_RATE_DECAY_COEF

input()
sim_gym_env(actor_net, env_name, num_timesteps=MAX_SIM_STEPS)