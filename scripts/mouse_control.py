#! /usr/bin/env python3
"""Agent that executes random actions"""
# stdlib
# import gym
from pathlib import Path

# third party
from click._unicodefun import click
import ipdb
import numpy as np

# first party
from hsr.hsr import MoveGripperEnv, print1
from mujoco import ObjType
from sac.util import make_box
from scripts.hsr import mutate_xml

saved_pos = None


@click.command()
@click.option('--xml-file', type=Path, default='world.xml')
@click.option('--discrete', is_flag=True)
def cli(discrete, xml_file):
    # env = NavigateEnv(continuous=True, max_steps=1000, geofence=.5)
    # env = Arm2PosEnv(action_multiplier=.01, history_len=1, continuous=True, max_steps=9999999, neg_reward=True)
    # env = Arm2TouchEnv(action_multiplier=.01, history_len=1, continuous=True, max_steps=9999999, neg_reward=True)
    # env = PickAndPlaceEnv(max_steps=9999999)
    goal_space = make_box((-.1, .1), (-.2, .2), (.418, .498))
    xml_filepath = Path(
        Path(__file__).parent.parent, 'environments', 'models', 'world.xml').absolute()
    with mutate_xml(
            changes=[],
            dofs=[
                "slide_x", "slide_y", "arm_lift_joint", "arm_flex_joint",
                "wrist_roll_joint", "hand_l_proximal_joint", "hand_r_proximal_joint"
            ],
            n_blocks=0,
            goal_space=goal_space,
            xml_filepath=xml_filepath) as temp_path:
        # xml_filepath = Path(Path(__file__).parent.parent, 'environments', 'models', xml_file) as temp_path:
        env = MoveGripperEnv(
            xml_filepath=temp_path,
            steps_per_action=300,
            geofence=.05,
            block_space=make_box((-.1, .1), (-.2, .2), (0, 1), (-1, 1)),
            goal_space=goal_space,
            randomize_pose=False,

            # fixed_goal=np.array([.11, .22, .4]),
            # min_lift_height=.43,
        )
    np.set_printoptions(precision=3, linewidth=800)
    env.reset()

    shape, = env.action_space.shape

    i = 0
    action = 0 if discrete else np.zeros(shape)
    moving = True
    pause = False
    done = False
    total_reward = 0
    s1 = env.reset()

    while True:
        lastkey = env.sim.get_last_key_press()
        if moving:
            if discrete:
                for k in range(1, 7):
                    if lastkey == str(k):
                        action = int(lastkey)

            else:
                action[i] += env.sim.get_mouse_dy() * .05

        if lastkey is 'R':
            print(env.reset())

        if lastkey is ' ':
            moving = not moving
            print('\nmoving:', moving)
        if lastkey is 'P':
            eu = env.unwrapped
            block_pos = eu.block_pos()
            print('\n')
            low = eu.goal().block - eu.goal_size / 2
            print('low', low)
            print('pos', block_pos)
            high = eu.goal().block + eu.goal_size / 2
            print('high', high)
            print('in between', (low <= block_pos) * (block_pos <= high))
            ipdb.set_trace()
            # print('gipper pos', env.env.gripper_pos())
            # for joint in [
            #         'slide_x', 'slide_y', 'arm_flex_joint', 'wrist_roll_joint',
            #         'hand_l_proximal_joint'
            # ]:
            #     print(joint, env.env.sim.qpos[env.env.sim.jnt_qposadr(joint)])
        # self.init_qpos[[self.sim.jnt_qposadr('slide_x'),
        #                 self.sim.jnt_qposadr('slide_y'),
        #                 self.sim.jnt_qposadr('arm_flex_joint'),
        #                 self.sim.jnt_qposadr('wrist_roll_joint'),
        #                 self.sim.jnt_qposadr('hand_l_proximal_joint'),
        #                 ]] = np.random.uniform(low=[-.13, -.23, -63, -90, 0],
        #                                        high=[.23, .25, 0, 90, 20])

        if not discrete:
            for k in range(10):
                if lastkey == str(k):
                    i = k - 1
                    print('')
                    print(env.sim.id2name(ObjType.ACTUATOR, i))

        if not pause and not np.allclose(action, 0):
            if not discrete:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            s2, r, done, _ = env.step(action)
            joints = 'slide_x slide_y arm_lift_joint arm_flex_joint wrist_roll_joint ' \
                     'hand_l_proximal_joint'.split()
            # 'wrist_flex_joint ' \
            joint_angles = [env.sim.get_joint_qpos(j) for j in joints]
            # print1(','.join(map(str, joint_angles)))
            print1(r)

            if discrete:
                action = 0

            total_reward += r
            # run_tests(env, s2)

        if done:
            if not pause:
                print('\nresetting', total_reward)
            total_reward = 0
            env.reset()
        # labels = {f'x{i}': g for i, g in
        #           enumerate([(x, y, z)
        #                      for x in env.env.goal_x
        #                      for y in env.env.goal_y
        #                      for z in env.env.goal_z])}
        env.render()


def run_tests(env, obs):
    assert env.env.observation_space.contains(obs)
    assert not env.env._currently_failed()
    assert np.shape(env._goal()) == np.shape(env.obs_to_goal(obs))
    goal, obs_history = env.destructure_mlp_input(obs)
    assert_equal(env._goal(), goal)
    assert_equal(env._get_obs(), obs_history[-1])
    assert_equal((goal, obs_history),
                 env.destructure_mlp_input(env.mlp_input(goal, obs_history)))
    assert_equal(obs, env.mlp_input(*env.destructure_mlp_input(obs)))
    assert_equal(obs, env.change_goal(goal, obs))
    try:
        assert_equal(env.gripper_pos(), env.gripper_pos(env.sim.qpos), atol=1e-2)
    except AttributeError:
        pass


def assert_equal(val1, val2, atol=1e-5):
    try:
        for a, b in zip(val1, val2):
            assert_equal(a, b, atol=atol)
    except TypeError:
        assert np.allclose(val1, val2, atol=atol), "{} vs. {}".format(val1, val2)


if __name__ == '__main__':
    cli()
