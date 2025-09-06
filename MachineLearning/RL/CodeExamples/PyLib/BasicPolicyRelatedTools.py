import numpy as np
import tensorflow as tf

class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float = 0.1):
        self.set_epsilon(epsilon)
    
    def set_epsilon(self, epsilon: float) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self._epsilon = epsilon
    
    def get_epsilon(self) -> float:
        return self._epsilon
    
    def get_action(self, q_values: np.ndarray) -> int:
        if len(q_values) == 0:
            raise ValueError("q_values cannot be empty")
            
        # Exploration: random action
        if np.random.random() < self._epsilon:
            return np.random.randint(len(q_values))
        
        # Exploitation: greedy action with random tie-breaking
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
    
    def get_action_probability(self, q_values: np.ndarray, action: int) -> float:
        if len(q_values) == 0:
            raise ValueError("q_values cannot be empty")
        if not 0 <= action < len(q_values):
            raise ValueError(f"action must be in [0, {len(q_values)-1}], got {action}")
        
        n_actions = len(q_values)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        n_best = len(best_actions)
        
        # Probability calculation
        if action in best_actions:
            return (1 - self._epsilon)/n_best + self._epsilon/n_actions
        else:
            return self._epsilon/n_actions

class SoftMaxPolicy:
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def get_action(self, action_prefs: np.ndarray) -> int:
        if len(action_prefs) == 0:
            raise ValueError("action_prefs cannot be empty")
        
        # Numerical stability trick
        action_prefs_norm = action_prefs - np.max(action_prefs)
        exp_h = np.exp(action_prefs_norm / self.temperature)
        action_probs = exp_h / np.sum(exp_h)
        
        return np.random.choice(len(action_prefs), p=action_probs)

    def get_action_probability(self, action_prefs: np.ndarray, action) -> np.ndarray:
        if len(action_prefs) == 0:
            raise ValueError("action_prefs cannot be empty")
        
        # Numerical stability trick
        action_prefs_norm = action_prefs - np.max(action_prefs)
        exp_h = np.exp(action_prefs_norm / self.temperature)
        return exp_h[action] / np.sum(exp_h)
    
    def get_probabilities(self, action_prefs: np.ndarray) -> np.ndarray:
        if len(action_prefs) == 0:
            raise ValueError("action_prefs cannot be empty")
        
        # Numerical stability trick
        action_prefs_norm = action_prefs - np.max(action_prefs)
        exp_h = np.exp(action_prefs_norm / self.temperature)
        return exp_h / np.sum(exp_h)
    
    def get_probabilities_tf(self, action_prefs: np.ndarray) -> np.ndarray:
        if len(action_prefs) == 0:
            raise ValueError("action_prefs cannot be empty")
        
        # Numerical stability trick
        action_prefs_norm = action_prefs - tf.reduce_max(action_prefs, 1, keepdims=True)
        exp_h = tf.exp(action_prefs_norm / self.temperature)
        return exp_h / tf.reduce_sum(exp_h)
    
    def get_action_probabilities_tf(self, action_prefs, actions) -> np.ndarray:
        if len(action_prefs) == 0:
            raise ValueError("action_prefs cannot be empty")
        
        # Numerical stability trick
        action_prefs_single_col = []
        i = 0
        for action in actions:
            action_prefs_single_col.append(action_prefs[i, int(action.numpy())])
            i += 1
        action_prefs_single_col = tf.reshape(tf.convert_to_tensor(action_prefs_single_col), (-1, 1))
        c = tf.reduce_max(action_prefs, 1, keepdims=True)
        action_prefs_norm = action_prefs_single_col - c
        exp_h = tf.exp((action_prefs - c) / self.temperature)
        return tf.exp(action_prefs_norm / self.temperature) / tf.reduce_sum(exp_h, 1, keepdims=True)
