import numpy as np

class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float = 0.1):
        """
        Initialize the epsilon-greedy policy.
        
        Args:
            epsilon: Exploration rate (0 ≤ epsilon ≤ 1)
        """
        self.set_epsilon(epsilon)
    
    def set_epsilon(self, epsilon: float) -> None:
        """
        Set the exploration rate.
        
        Args:
            epsilon: New exploration rate (0 ≤ epsilon ≤ 1)
            
        Raises:
            ValueError: If epsilon is outside [0, 1]
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self._epsilon = epsilon
    
    def get_epsilon(self) -> float:
        """
        Get the current exploration rate.
        
        Returns:
            Current epsilon value
        """
        return self._epsilon
    
    def get_action(self, q_values: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            q_values: 1D array of Q-values for current state (shape: [n_actions,])
            
        Returns:
            Selected action index (int)
            
        Raises:
            ValueError: If q_values is empty
        """
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
        """
        Get the probability of selecting a specific action.
        
        Args:
            q_values: 1D array of Q-values for current state (shape: [n_actions,])
            action: Action index to get probability for
            
        Returns:
            Probability of selecting the given action
            
        Raises:
            ValueError: If q_values is empty or action is invalid
        """
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