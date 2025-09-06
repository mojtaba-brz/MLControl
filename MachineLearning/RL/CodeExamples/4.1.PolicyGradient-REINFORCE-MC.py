import gymnasium as gym
import tensorflow as tf

from PyLib.ActorCritic import ActorNetwork, CriticNetwork, actor_network_default_dict
from PyLib.RenderTools import render_gym_env, sim_gym_env

# Algorithm:
#   Loop forever:
#       Generate an episode
#       Loop for each step if the episode t=0:T-1
#           G = \sum{k=t+1}{T} gamma^(k-t-1) * r_k
#           theta = theta + alpha * gamma^t * G grad(ln(pi(a_t|s_t, theta)))

env_name = "CartPole-v1"
# env_name = "Acrobot-v1"
# env_name = "MountainCar-v0"
# env_name = "LunarLander-v3"

env = gym.make(env_name)

MAX_EPISODES = 100000
GAMMA = 0.99
SHOW_SIM_EVERY_N_EPISODE = 5000000
INITIAL_LEARNING_RATE = 2e-3
LEARNING_RATE_DECAY_COEF = 0.999
learning_rate = INITIAL_LEARNING_RATE #An improvement mentioned by sutton in his RL book

actor_params = actor_network_default_dict
actor_params['is_deterministic'] = False
actor_params['action_space_is_discrete'] = True
actor_params['soft_max_policy_temperature'] = 1.0
actor_params['hidden_layer_sizes'] = (128, )
actor_params['state_shape'] = env.observation_space.shape[0]
actor_params['action_shape'] = env.action_space.n
actor_params['optimizer'] = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
actor_params['ppo_epsilon'] = 0.05
actor_params['value_gamma'] = GAMMA

actor_net     = ActorNetwork(actor_params)

steps = 0
actor_loss = None
ep_reward_filt = 0
for ep in range(MAX_EPISODES):
    state, info = env.reset()
    done = False
    ep_reward = 0

    reward_array = []
    action_array = []
    state_array = []

    while not done:
        action = actor_net.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward = reward if not done else -200
        ep_reward += reward

        reward_array.append(reward)
        action_array.append(action)
        state_array.append(state)

        state = next_state
        steps += 1
    
    reward_array_tf = tf.convert_to_tensor(reward_array, dtype=tf.float32)
    action_array_tf = tf.convert_to_tensor(action_array, dtype=tf.float32)
    state_array_tf = tf.convert_to_tensor(state_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        actor_loss = actor_net.calc_pg_mc_loss(reward_array_tf, state_array_tf, action_array_tf)
    actor_grads = tape.gradient(actor_loss, actor_net.trainable_variables)
    actor_net.update(actor_grads)
    
    steps = 0
    reward_array = []
    action_array = []
    state_array = []

    if ep>0 and ep%SHOW_SIM_EVERY_N_EPISODE==0:
        sim_gym_env(actor_net, env_name, num_timesteps=500)

    ep_reward_filt += 0.1*(ep_reward - ep_reward_filt)
    if actor_loss is not None and ep%10==0:
        print(f"Episode: {ep})\tReward: {ep_reward_filt:5.0f}, {ep_reward:5.0f}\tActor Loss: {actor_loss:.5f}")
        learning_rate *= LEARNING_RATE_DECAY_COEF
        actor_net.optimizer._learning_rate = (learning_rate)

input()
sim_gym_env(actor_net, env_name, num_timesteps=500)