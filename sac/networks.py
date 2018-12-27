# stdlib
from typing import Tuple

# third party
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple

# first party
from sac.agent import AbstractAgent, NetworkOutput
from sac.util import ArrayLike, Step


class SACXAgent(AbstractAgent):
    @property
    def seq_len(self):
        return None

    @staticmethod
    def actor_network(inputs: tf.Tensor) -> NetworkOutput:
        l1 = tf.layers.dense(inputs, 200, tf.nn.elu)
        l2 = tf.contrib.layers.layer_norm(l1)
        l3 = tf.layers.dense(l2, 200, tf.nn.elu)
        return NetworkOutput(output=tf.layers.dense(l3, 100, tf.nn.elu), state=None)

    @staticmethod
    def critic_network(inputs: tf.Tensor) -> NetworkOutput:
        l1 = tf.layers.dense(inputs, 400, tf.nn.elu)
        l2 = tf.contrib.layers.layer_norm(l1)
        l3 = tf.layers.dense(l2, 400, tf.nn.elu)
        return NetworkOutput(output=tf.layers.dense(l3, 200, tf.nn.elu), state=None)

    def pi_network(self, o: tf.Tensor):
        with tf.variable_scope('pi'):
            return self.actor_network(o)

    def getQ(self, o: tf.Tensor, a: tf.Tensor, name: str, reuse: bool = None):
        with tf.variable_scope(name, reuse=reuse):
            oa = tf.concat([o, a], axis=1)
            return tf.reshape(
                tf.layers.dense(self.critic_network(oa).output, 1, name='q'), [-1])

    def v_network(self, o: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return tf.reshape(
                tf.layers.dense(self.critic_network(o).output, 1, name='v'), [-1])


class LstmAgent(AbstractAgent):
    @property
    def seq_len(self):
        if self._seq_len:
            return self._seq_len
        return 8

    def __init__(self, batch_size: int, layer_size: int, device_num: int, **kwargs):
        self.batch_size = batch_size
        self.layer_size = layer_size
        with tf.device('/gpu:' + str(device_num)):
            state_args = tf.float32, [batch_size, layer_size]
            self.S = LSTMStateTuple(
                c=tf.placeholder(*state_args, name='C'),
                h=tf.placeholder(*state_args, name='H'))
            self.lstm = BasicLSTMCell(layer_size)
        super().__init__(layer_size=layer_size, device_num=device_num, **kwargs)
        self.initial_state = self.sess.run(self.lstm.zero_state(batch_size, tf.float32))
        assert np.shape(self.initial_state) == (2, batch_size, layer_size)
        assert self.S.c.shape == self.S.h.shape == (batch_size, layer_size)

    def network(self, inputs: tf.Tensor, reuse=False) -> tf.Tensor:
        split_inputs = tf.split(inputs, self.seq_len, axis=1)
        s = self.S
        for x in split_inputs:
            x = tf.squeeze(x, axis=1)
            outputs = NetworkOutput(*self.lstm(x, s))
        return outputs

    def state_feed(self, states):
        return dict(zip(self.S, states))

    def train_step(self, step: Step) -> dict:
        assert np.shape(step.s) == np.shape(self.initial_state)
        if feed_dict is None:
            feed_dict = {
                **self.state_feed(step.s),
                **{
                    self.o1: step.o1,
                    self.a:  step.a,
                    self.r:  np.array(step.r) * self.reward_scale,
                    self.o2: step.o2,
                    self.t:  step.t
                }
            }
        return super().train_step(step)

    def getQ(self, o: tf.Tensor, a: tf.Tensor, name: str, reuse: bool = None) \
            -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            o = self.network(o).output
            oa = tf.concat([o, a], axis=1)
            return tf.reshape(tf.layers.dense(oa, 1, name='q'), [-1])

    def get_actions(self, o: ArrayLike, sample: bool = True, state=None) \
            -> Tuple[np.ndarray, LSTMStateTuple]:
        assert len(np.shape(o)) == 1
        assert np.shape(state) == np.shape(self.initial_state)
        feed_dict = {**{self.o1: [[o]]}, **self.state_feed(state)}
        A = self.A_sampled1 if sample else self.A_max_likelihood
        return self.sess.run([A[0], self.S_new], feed_dict)


class MoEAgent(AbstractAgent):
    def __init__(self, n_networks, device_num, **kwargs):
        self.n_networks = n_networks
        super().__init__(device_num=device_num, **kwargs)

    @property
    def seq_len(self):
        return None

    def network(self, inputs: tf.Tensor):
        with tf.variable_scope('weights'):
            weights = mlp(
                inputs,
                self.layer_size,
                n_layers=self.n_layers - 1,
                activation=self.activation)
            weights = tf.layers.dense(weights, units=self.n_networks)
            weights = tf.nn.softmax(logits=weights, axis=-1)
        weights = tf.expand_dims(weights, axis=1)  # [batch, hidden, networks]

        def vote(i):
            with tf.variable_scope('vote' + str(i)):
                return mlp(inputs, self.layer_size, self.n_layers - 1, self.activation)

        h = tf.stack(
            values=list(map(vote, range(self.n_networks))),
            axis=-1)  # [batch, hidden, networks]
        h = tf.reduce_sum(h * weights, axis=2)
        output = tf.layers.dense(h, units=self.layer_size, name='output')
        return NetworkOutput(output=output, state=weights)
