# third party
import click
from gym.wrappers import TimeLimit
import tensorflow as tf

# first party
from environments.frozen_lake import FrozenLakeEnv
from environments.hierarchical_wrapper import FrozenLakeHierarchicalWrapper
from sac.hierarchical_trainer import HierarchicalTrainer
from sac.networks import MLPAgent
from sac.train import Trainer
from sac.utils import create_sess


def check_probability(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter("Param {} should be between 0 and 1".format(value))
    return value


def parse_double(ctx, param, string):
    a, b = map(int, string.split('x'))
    return a, b


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--n-layers', default=2, type=int)
@click.option('--layer-size', default=128, type=int)
@click.option('--learning-rate', default=1e-4, type=float)
@click.option('--buffer-size', default=1e5, type=int)
@click.option('--num-train-steps', default=1, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=1, type=float)
@click.option('--entropy-scale', default=5e-6, type=float)
@click.option('--grad-clip', default=None, type=float)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--map-dims', default="4x4", type=str, callback=parse_double)
@click.option('--random-map', is_flag=True)
@click.option('--random-start', is_flag=True)
@click.option('--random-goal', is_flag=True)
@click.option('--default-reward', default=-.1, type=float)
@click.option('--is-slippery', is_flag=True)
@click.option('--max-steps', default=100, type=int)
@click.option('--render', is_flag=True)
@click.option('--boss-oracle', is_flag=True)
@click.option('--worker-oracle', is_flag=True)
@click.option('--boss-freq', default=None, type=int)
def cli(seed, buffer_size, n_layers, layer_size, learning_rate, entropy_scale,
        reward_scale, batch_size, num_train_steps, logdir, save_path, load_path, render,
        grad_clip, map_dims, max_steps, random_map, random_start, random_goal,
        is_slippery, default_reward, boss_freq, worker_oracle, boss_oracle):
    env = TimeLimit(
        env=FrozenLakeEnv(
            map_dims=map_dims,
            random_map=random_map,
            random_start=random_start,
            random_goal=random_goal,
            is_slippery=is_slippery,
            default_reward=default_reward,
        ),
        max_episode_steps=max_steps)

    kwargs = dict(
        sess=create_sess(),
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
        grad_clip=grad_clip,
        num_train_steps=num_train_steps,
    )
    if boss_freq:
        trainer = HierarchicalTrainer(
            boss_act_freq=boss_freq,
            use_worker_oracle=worker_oracle,
            use_boss_oracle=boss_oracle,
            env=FrozenLakeHierarchicalWrapper(env),
            **kwargs)
    else:
        trainer = Trainer(env=env, **kwargs)

    trainer.train(load_path=load_path, logdir=logdir, render=render, save_path=save_path)


if __name__ == '__main__':
    cli()
