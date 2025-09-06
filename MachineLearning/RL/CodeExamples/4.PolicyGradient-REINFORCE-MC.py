import gymnasium as gym
import tensorflow as tf
import numpy as np

from CodeExamples.PyLib.ActorCritic import ActorNetwork, CriticNetwork, actor_network_default_dict
from CodeExamples.PyLib.RenderTools import render_gym_env, sim_gym_env

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

MAX_EPISODES = 1000
# UPDATE_AFTER_N_STEPS = 40
GAMMA = 0.99
ACTOR_UPDATE_TIMES = 5
CRITIC_UPDATE_TIMES = 5
SHOW_SIM_EVERY_N_EPISODE = 5000000000

actor_params = actor_network_default_dict
actor_params['is_deterministic'] = False
actor_params['action_space_is_discrete'] = True
actor_params['soft_max_policy_temperature'] = 1.0
actor_params['hidden_layer_sizes'] = (100, 100)
actor_params['state_shape'] = env.observation_space.shape[0]
actor_params['action_shape'] = env.action_space.n
actor_params['optimizer'] = tf.keras.optimizers.Adam(learning_rate=0.0003)
actor_params['ppo_epsilon'] = 0.05
actor_params['value_gamma'] = GAMMA

critic_params = { 'hidden_layer_sizes':(256, 0),
                  'output_layer_activation_function':'linear',
                  'hidden_layers_activation_function':'relu',
                  'state_shape':env.observation_space.shape[0],
                  'mode':'V',
                  'optimizer':tf.keras.optimizers.Adam(learning_rate=0.1),}

actor_net     = ActorNetwork(actor_params)
actor_net_old = ActorNetwork(actor_params)
actor_net_old.set_weights(actor_net.get_weights())
value_net     = CriticNetwork(critic_params)

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
    next_state_array = []

    while not done:
        action = actor_net.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward = reward if not done else -200
        ep_reward += reward

        reward_array.append(reward)
        action_array.append(action)
        state_array.append(state)
        next_state_array.append(next_state)

        state = next_state
        steps += 1
        if steps >= UPDATE_AFTER_N_STEPS:
            reward_array_tf = tf.convert_to_tensor(reward_array, dtype=tf.float32)
            action_array_tf = tf.convert_to_tensor(action_array, dtype=tf.float32)
            state_array_tf = tf.convert_to_tensor(state_array, dtype=tf.float32)
            next_state_array_tf = tf.convert_to_tensor(next_state_array, dtype=tf.float32)

            next_values = tf.squeeze(value_net(next_state_array_tf), axis=1)
            values = tf.squeeze(value_net(state_array_tf), axis=1)
            targets = reward_array_tf + GAMMA*next_values
            actor_weights_before_update = actor_net.get_weights()
            for _ in range(ACTOR_UPDATE_TIMES):
                with tf.GradientTape() as tape:
                    actor_loss = actor_net.calc_pg_loss(targets-values, state_array_tf, action_array_tf)
                actor_grads = tape.gradient(actor_loss, actor_net.trainable_variables)
                actor_net.update(actor_grads)
            actor_net_old.set_weights(actor_weights_before_update)

            for _ in range(CRITIC_UPDATE_TIMES):
                with tf.GradientTape() as tape:
                    values = tf.squeeze(value_net(state_array_tf), axis=1)
                    err = targets - values
                    value_loss = tf.reduce_mean(tf.square(err))
                critic_grads = tape.gradient(value_loss, value_net.trainable_variables)
                value_net.update(critic_grads)
            steps = 0
            reward_array = []
            action_array = []
            state_array = []
            next_state_array = []

    if ep>0 and ep%SHOW_SIM_EVERY_N_EPISODE==0:
        sim_gym_env(actor_net, env_name, num_timesteps=500)

    ep_reward_filt += 0.1*(ep_reward - ep_reward_filt)
    if actor_loss is not None and ep%5==0:
        print(f"Episode: {ep})\tReward: {ep_reward_filt:5.0f}, {ep_reward:5.0f}\tActor Loss: {actor_loss:.5f}\tValue Loss: {value_loss:.5f}")

sim_gym_env(actor_net, env_name, num_timesteps=500)