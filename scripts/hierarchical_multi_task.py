# stdlib
from pathlib import Path

# third party
import click
from gym.wrappers import TimeLimit
import tensorflow as tf

# first party
from environments.hierarchical_wrapper import (FrozenLakeHierarchicalWrapper, HierarchicalWrapper,
                                               ShiftHierarchicalWrapper)
from environments.shift import ShiftEnv
from sac.hierarchical_trainer import HierarchicalTrainer
from sac.networks import MLPAgent
from sac.utils import create_sess
from scripts.lift import env_wrapper, put_in_xml_setter
from scripts.shift import parse_coordinate


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
@click.option('--max-steps', default=100, type=int)
@click.option('--steps-per-action', default=200, type=int)
@click.option('--render', is_flag=True)
@click.option('--render-freq', default=0, type=int)
@click.option('--record', is_flag=True)
@click.option('--record-freq', type=int, default=0)
@click.option('--record-path', type=Path)
@click.option('--image-dims', type=str, default='800x800', callback=parse_double)
@click.option('--randomize-pose', is_flag=True)
@click.option('--set-xml', multiple=True, callback=put_in_xml_setter)
@click.option('--geofence', default=.25, type=float)
@click.option('--hindsight-geofence', default=None, type=float)
@click.option('--fixed-block', default=None, callback=parse_coordinate)
@click.option('--fixed-goal', default=None, callback=parse_coordinate)
@click.option('--goal-x', default=None, callback=parse_coordinate)
@click.option('--goal-y', default=None, callback=parse_coordinate)
@click.option('--xml-file', type=Path, default='world.xml')
@click.option('--correct-boss-action', is_flag=True)
@click.option('--boss-oracle', is_flag=True)
@click.option('--worker-oracle', is_flag=True)
@click.option('--boss-freq', default=None, type=int)
@click.option(
    '--use-dof',
    multiple=True,
    default=[
        'slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint', 'wrist_roll_joint',
        'hand_l_proximal_joint', 'hand_r_proximal_joint', 'goal_x', 'goal_y'
    ])
@env_wrapper
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
        render_freq,
        max_steps,
        steps_per_action,
        boss_freq,
        worker_oracle,
        boss_oracle,
        correct_boss_action,
        record,
        record_freq,
        record_path,
        image_dims,
        randomize_pose,
        geofence,
        hindsight_geofence,
        fixed_block,
        fixed_goal,
        goal_x,
        goal_y,
        temp_path,
):
    env = TimeLimit(
        max_episode_steps=max_steps,
        env=ShiftEnv(
            geofence=geofence,
            xml_filepath=temp_path,
            steps_per_action=steps_per_action,
            render_freq=render_freq,
            record=record,
            record_path=record_path,
            record_freq=record_freq,
            image_dimensions=image_dims,
            fixed_block=fixed_block,
            fixed_goal=fixed_goal,
            randomize_pose=randomize_pose,
            goal_x=goal_x,
            goal_y=goal_y,
        ))

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

    HierarchicalTrainer(
        env=ShiftHierarchicalWrapper(env, geofence=geofence),
        boss_act_freq=boss_freq,
        use_worker_oracle=worker_oracle,
        use_boss_oracle=boss_oracle,
        worker_kwargs=worker_kwargs,
        boss_kwargs=boss_kwargs,
        **kwargs).train(
            load_path=load_path, logdir=logdir, render=False, save_path=save_path)


if __name__ == '__main__':
    cli()
