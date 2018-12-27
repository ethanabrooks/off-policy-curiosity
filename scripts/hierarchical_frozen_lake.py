# third party
import click
from gym.wrappers import TimeLimit
import tensorflow as tf

# first party
from environments.frozen_lake import FrozenLakeEnv
from environments.hierarchical_wrapper import FrozenLakeHierarchicalWrapper
from sac.hierarchical_trainer import HierarchicalTrainer
from sac.networks import MLPAgent
from sac.util import create_sess


def check_probability(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter("Param {} should be between 0 and 1".format(value))
    return value


def parse_double(ctx, param, string):
    a, b = map(int, string.split('x'))
    return a, b


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--worker-n-layers', default=2, type=int)
@click.option('--worker-layer-size', default=256, type=int)
@click.option('--worker-learning-rate', default=3e-3, type=float)
@click.option('--worker-reward-scale', default=1, type=float)
@click.option('--worker-entropy-scale', default=2e-7, type=float)
@click.option('--worker-num-train-steps', default=1, type=int)
@click.option('--worker-grad-clip', default=None, type=float)
@click.option('--boss-n-layers', default=2, type=int)
@click.option('--boss-layer-size', default=256, type=int)
@click.option('--boss-learning-rate', default=2e-3, type=float)
@click.option('--boss-reward-scale', default=1, type=float)
@click.option('--boss-entropy-scale', default=5e-8, type=float)
@click.option('--boss-num-train-steps', default=1, type=int)
@click.option('--boss-grad-clip', default=1, type=float)
@click.option('--buffer-size', default=1e5, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--map-dims', default="4x4", type=str, callback=parse_double)
@click.option('--random-map', is_flag=True)
@click.option('--random-start', is_flag=True)
@click.option('--random-goal', is_flag=True)
@click.option('--default-reward', default=-.1, type=float)
@click.option('--is-slippery', is_flag=True)
@click.option('--max-steps', default=12, type=int)
@click.option('--render', is_flag=True)
@click.option('--correct-boss-action', is_flag=True)
@click.option('--boss-oracle', is_flag=True)
@click.option('--worker-oracle', is_flag=True)
@click.option('--boss-freq', default=None, type=int)
@click.option('--n-boss-actions', default=None, type=int)
def cli(
        seed,
        buffer_size,
        worker_n_layers,
        worker_layer_size,
        worker_learning_rate,
        worker_entropy_scale,
        worker_reward_scale,
        worker_num_train_steps,
        worker_grad_clip,
        boss_n_layers,
        boss_layer_size,
        boss_learning_rate,
        boss_entropy_scale,
        boss_reward_scale,
        boss_num_train_steps,
        boss_grad_clip,
        batch_size,
        logdir,
        save_path,
        load_path,
        render,
        map_dims,
        max_steps,
        random_map,
        random_start,
        random_goal,
        is_slippery,
        default_reward,
        boss_freq,
        n_boss_actions,
        worker_oracle,
        boss_oracle,
        correct_boss_action,
):

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
        batch_size=batch_size,
    )
    worker_kwargs = dict(
        n_layers=worker_n_layers,
        layer_size=worker_layer_size,
        learning_rate=worker_learning_rate,
        entropy_scale=worker_entropy_scale,
        reward_scale=worker_reward_scale,
        grad_clip=worker_grad_clip,
        num_train_steps=worker_num_train_steps,
    )

    boss_kwargs = dict(
        n_layers=boss_n_layers,
        layer_size=boss_layer_size,
        learning_rate=boss_learning_rate,
        entropy_scale=boss_entropy_scale,
        reward_scale=boss_reward_scale,
        grad_clip=boss_grad_clip,
        num_train_steps=boss_num_train_steps,
    )

    # n_boss_actions = (1 + 2 * boss_freq) ** 2
    HierarchicalTrainer(
        env=FrozenLakeHierarchicalWrapper(env, n_boss_actions=n_boss_actions),
        boss_act_freq=boss_freq,
        use_worker_oracle=worker_oracle,
        use_boss_oracle=boss_oracle,
        worker_kwargs=worker_kwargs,
        boss_kwargs=boss_kwargs,
        **kwargs).train(
            load_path=load_path, logdir=logdir, render=render, save_path=save_path)


if __name__ == '__main__':
    cli()
