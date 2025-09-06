from queue import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.memory = deque(maxlen=size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
    def store(self, obs, act, rew, next_obs, done):
        self.memory.append((obs, act, rew, next_obs, done))
        
    def sample_batch(self, batch_size=32):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        
        # Convert batch of tuples to tuple of batches
        s_batch, a_batch, r_batch, s2_batch, d_batch = zip(*batch)
        
        return dict(
            s=np.array(s_batch, dtype=np.float32),
            s2=np.array(s2_batch, dtype=np.float32),
            a=np.array(a_batch, dtype=np.float32),
            r=np.array(r_batch, dtype=np.float32),
            d=np.array(d_batch, dtype=np.float32)
        )
    
    @property
    def size(self):
        return len(self.memory)


class ReplayBuffer2:
    def __init__(self, max_size, input_shape, n_actions):
        self.m_size = max_size
        self.m_cntr = 0
        self.state_m = np.zeros((self.m_size, *input_shape))
        self.new_state_m = np.zeros((self.m_size, *input_shape))
        self.action_m = np.zeros((self.m_size, n_actions))
        self.reward_m = np.zeros(self.m_size)
        self.terminal_m = np.zeros(self.m_size)

    def store(self, state, action, reward, state_, done):
        id = self.m_cntr % self.m_size

        self.state_m[id] = state
        self.new_state_m[id] = state_
        self.action_m[id] = action
        self.reward_m[id] = reward
        self.terminal_m[id] = done

        self.m_cntr += 1

    def sample(self, batch_size):
        max_m = min(self.m_cntr, self.m_size)

        batch = np.random.choice(max_m, batch_size)

        states = self.state_m[batch]
        states_ = self.new_state_m[batch]
        actions = self.action_m[batch]
        rewards = self.reward_m[batch]
        dones = self.terminal_m[batch]

        return states, actions, rewards, states_, dones
    

class ReplayBuffer3:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def store(self, state, action, reward, next_state, done):
        # Convert everything to float32 when storing
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.float32(reward)  # Convert scalar to float32
        next_state = np.array(next_state, dtype=np.float32)
        done = np.float32(done)  # Convert boolean to float32
        
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in indices:
            s, a, r, s2, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s2)
            dones.append(d)

        return {
            's': np.array(states, dtype=np.float32),
            'a': np.array(actions, dtype=np.float32),
            'r': np.array(rewards, dtype=np.float32),
            's2': np.array(next_states, dtype=np.float32),
            'd': np.array(dones, dtype=np.float32)
        }
    
    def __len__(self):
        return len(self.buffer)