# third party
import tensorflow as tf

from sac.agent import AbstractAgent
from sac.utils import make_network

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
        super().__init__(network_args=network_args, o_size=o_size, a_size=a_size,
                         **kwargs)

    def produce_policy_parameters(self, a_size: int, o1: tf.Tensor):
        processed_s = self.pi_network(o1)
        mu, sigma_param = tf.split(processed_s, 2 , axis=1)
        return mu, tf.sigmoid(sigma_param) + 0.0001

    @staticmethod
    def policy_parameters_to_log_prob(u, parameters):
        (mu, sigma) = parameters
        log_prob = tf.distributions.Normal(mu, sigma).log_prob(u)
        # print(log_prob)
        return tf.reduce_sum(
            log_prob, axis=1) - tf.reduce_sum(
            tf.log(1 - tf.square(tf.tanh(u)) + EPS), axis=1)

    @staticmethod
    def policy_parameters_to_max_likelihood_action(parameters):
        (mu, sigma) = parameters
        return mu

    @staticmethod
    def policy_parameters_to_sample(parameters):
        (mu, sigma) = parameters
        return tf.distributions.Normal(mu, sigma).sample()

    @staticmethod
    def transform_action_sample(action_sample):
        return tf.tanh(action_sample)

    @staticmethod
    def entropy_from_params(parameters):
        (mu, sigma) = parameters
        return tf.distributions.Normal(mu, sigma).entropy()


class GaussianMixturePolicy(object):
    def produce_policy_parameters(self, a_shape, processed_s):
        pass

    def policy_parmeters_to_log_prob(self, a, parameters):
        pass

    def policy_parameters_to_sample(self, parameters):
        pass


class CategoricalPolicy(AbstractAgent):
    @staticmethod
    def produce_policy_parameters(a_shape, processed_s):
        logits = tf.layers.dense(processed_s, a_shape, name='logits')
        return logits

    @staticmethod
    def policy_parameters_to_log_prob(a, parameters):
        logits = parameters
        out = tf.distributions.Categorical(logits=logits).log_prob(tf.argmax(a, axis=1))
        # out = tf.Print(out, [out], summarize=10)
        return out

    @staticmethod
    def policy_parameters_to_sample(parameters):
        logits = parameters
        a_shape = logits.get_shape()[1].value
        # logits = tf.Print(logits, [tf.nn.softmax(logits)], message='logits are:',
        # summarize=10)
        out = tf.one_hot(tf.distributions.Categorical(logits=logits).sample(), a_shape)
        return out

    @staticmethod
    def policy_parameters_to_max_likelihood_action(parameters):
        logits = parameters
        a_shape = logits.get_shape()[1].value
        return tf.one_hot(tf.argmax(logits, axis=1), a_shape)

    @staticmethod
    def transform_action_sample(action_sample):
        return action_sample

    @staticmethod
    def entropy_from_params(logits):
        return tf.distributions.Categorical(logits).entropy()
