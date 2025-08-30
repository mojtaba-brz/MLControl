import random
from collections import deque

import numpy as np
from tensorflow.keras import backend as K

from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Add, Subtract, Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

from .BasicPolicyRelatedTools import EpsilonGreedyPolicy

class DQNAgent:
    def __init__(self, state_size, action_size, alpha, use_double_learning = True, target_model_update_freq = 50,
                 use_dueling_net = False, use_PER = False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 
        self.learning_rate = alpha 
        self.train_start = 1000
        self.target_model_update_freq = target_model_update_freq
        self.target_model_update_count = 0
        self.target_model_is_active = use_double_learning
        self.use_dueling_net = use_dueling_net
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.policy = EpsilonGreedyPolicy(self.epsilon)
        
    def _build_model(self):
        initializer = initializers.GlorotNormal(seed=42)
        if self.use_dueling_net:
            state_input = Input((self.state_size,))
            x = Dense(8, activation='relu', kernel_initializer=initializer)(state_input)
            x = Dense(16, activation='relu', kernel_initializer=initializer)(x)
            value = Dense(1, activation='linear', kernel_initializer=initializer)(x)
            advantage = Dense(self.action_size, activation='linear', kernel_initializer=initializer)(x)
            mean_advantage = Lambda(lambda a: K.mean(a, axis=1, keepdims=True), output_shape=(1,))(advantage)
            q_values = Add()([value, Subtract()([advantage, mean_advantage])])
            model = Model(inputs=state_input, outputs=q_values)
        else:
            model = Sequential([
                Input((self.state_size, )),
                Dense(8, activation='relu', kernel_initializer=initializer),
                Dense(16, activation='relu', kernel_initializer=initializer),
                Dense(self.action_size, activation='linear', kernel_initializer=initializer)
            ])
            
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def act(self, state):
        q_values = self.model.predict(state, verbose=0)
        return self.policy.get_action(q_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return []

        minibatch = random.sample(self.memory, batch_size)
        batch_loss = []

        current_state = np.zeros((batch_size, self.state_size))
        next_state = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            current_state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # Standard DQN update
        target_q = self.model.predict(current_state, verbose=0)
        if self.target_model_is_active:
            next_q_values = self.target_model.predict(next_state, verbose=0)
        else:   
            next_q_values = self.model.predict(next_state, verbose=0)

        for i in range(batch_size):
            if done[i]:
                target_q[i][action[i]] = reward[i]
            else:
                target_q[i][action[i]] = reward[i] + self.gamma * np.max(next_q_values[i])

        hist = self.model.fit(current_state, target_q, batch_size=batch_size, epochs=1, verbose=0)
        batch_loss.append(hist.history['loss'][-1])

        if (self.epsilon > self.epsilon_min) and \
             (len(self.memory) >= self.train_start):
                self.epsilon *= self.epsilon_decay
                self.policy.set_epsilon(self.epsilon)

        if self.target_model_is_active:
            if self.target_model_update_count % self.target_model_update_freq == 0:
                self.update_target_network()
                self.target_model_update_count = 0
            else:
                self.target_model_update_count += 1

        return batch_loss
    