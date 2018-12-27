# third party
import tensorflow as tf

from sac.agent import AbstractAgent
from sac.util import make_network

EPS = 1E-6


class GaussianPolicy(AbstractAgent):
    """
    Policy outputs a gaussian action that is clamped to the interval [-1, 1]
    """

    def __init__(self, o_size: int, a_size: int, network_args: dict, **kwargs):
        self.network_args = network_args
        args = self.network_args.copy()
        args.update(n_hidden=args['n_hidden'] + 1)
        self.pi_network = make_network(o_size, 2 * a_size, **args)
        super().__init__(
            network_args=network_args, o_size=o_size, a_size=a_size, **kwargs)

    def get_policy_params(self, obs: tf.Tensor):
        processed_s = self.pi_network(obs)
        mu, sigma_param = tf.split(processed_s, 2, axis=1)
        return mu, tf.sigmoid(sigma_param) + 0.0001

    @staticmethod
    def policy_parameters_to_log_prob(action, parameters):
        (mu, sigma) = parameters
        log_prob = tf.distributions.Normal(mu, sigma).log_prob(action)
        return tf.reduce_sum(
            log_prob, axis=1) - tf.reduce_sum(
            tf.log(1 - tf.square(tf.tanh(action)) + EPS), axis=1)

    @staticmethod
    def policy_parameters_to_max_likelihood_action(parameters):
        (mu, sigma) = parameters
        return mu

    @staticmethod
    def policy_parameters_to_sample(parameters):
        (mu, sigma) = parameters
        return tf.distributions.Normal(mu, sigma).sample()

    @staticmethod
    def preprocess_action(action_sample):
        return tf.tanh(action_sample)

    @staticmethod
    def entropy_from_params(parameters):
        (mu, sigma) = parameters
        return tf.distributions.Normal(mu, sigma).entropy()


class GaussianMixturePolicy(AbstractAgent):
    def get_policy_params(self, obs):
        pass

    def policy_parmeters_to_log_prob(self, a, parameters):
        raise NotImplementedError

    def policy_parameters_to_sample(self, parameters):
        raise NotImplementedError


class CategoricalPolicy(AbstractAgent):
    def __init__(self, a_size, o_size, network_args, **kwargs):
        super().__init__(
            a_size=a_size, o_size=o_size, network_args=network_args, **kwargs)
        self.pi_network = make_network(
            o_size,
            a_size,
            **network_args,
        )

    def get_policy_params(self, obs):
        return self.pi_network(obs)

    @staticmethod
    def policy_parameters_to_log_prob(action, parameters):
        logits = parameters
        return tf.distributions.Categorical(logits=logits).log_prob(tf.argmax(action,
                                                                              axis=1))

    @staticmethod
    def policy_parameters_to_sample(parameters):
        logits = parameters
        a_shape = logits.get_shape()[1].value
        return tf.one_hot(tf.distributions.Categorical(logits=logits).sample(), a_shape)

    @staticmethod
    def policy_parameters_to_max_likelihood_action(parameters):
        logits = parameters
        a_shape = logits.get_shape()[1].value
        return tf.one_hot(tf.argmax(logits, axis=1), a_shape)

    @staticmethod
    def preprocess_action(action_sample):
        return action_sample

    @staticmethod
    def entropy_from_params(logits):
        return tf.distributions.Categorical(logits).entropy()
