import numpy as np
import gymnasium as gym
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class FeatureTransformer:
    # create 4 set of RFB networks that convert inputs of an ANN to features
    # TODO: Study the function and classes.
    # TODO: Make it more customizable
    def __init__(self, env : gym.wrappers.common.TimeLimit, 
                 components_gammas=((100, 5.0), (100, 2), (100, 1), (100, 0.5)), 
                 n_samples=1000):
        observation_examples = np.array([env.observation_space.sample() for _ in range(n_samples)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        feature_layers_list = []
        layer_counter = 0
        for component_gamma in components_gammas:
            layer_counter += 1
            component = component_gamma[0]
            gamma = component_gamma[1]
            feature_layers_list += [(f"rbf_{layer_counter}", RBFSampler(gamma=gamma, n_components=component))]
        featurizer = FeatureUnion(feature_layers_list)
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)