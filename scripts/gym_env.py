# third party
import click
import gym
import tensorflow as tf

# first party
from sac.networks import MLPAgent
from sac.train import Trainer


def check_probability(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter("Param {} should be between 0 and 1".format(value))
    return value


@click.command()
@click.option('--env', default='CartPole-v0')
@click.option('--seed', default=0, type=int)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=3e-4, type=float)
@click.option('--buffer-size', default=1e5, type=int)
@click.option('--num-train-steps', default=1, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=1., type=float)
@click.option('--entropy-scale', default=1., type=float)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render', is_flag=True)
def cli(env, seed, buffer_size, n_layers, layer_size, learning_rate, reward_scale,
        entropy_scale, batch_size, num_train_steps, logdir, save_path, load_path, render):
    Trainer(
        env=gym.make(env),
        base_agent=MLPAgent,
        seq_len=0,
        device_num=1,
        seed=seed,
        buffer_size=buffer_size,
        activation=tf.nn.relu,
        n_layers=n_layers,
        layer_size=layer_size,
        learning_rate=learning_rate,
        entropy_scale=entropy_scale,
        reward_scale=reward_scale,
        batch_size=batch_size,
        grad_clip=None,
        n_train_steps=num_train_steps)


if __name__ == '__main__':
    cli()
