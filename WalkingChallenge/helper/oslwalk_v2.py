""" =================================================
Modifications made by Adan Dominguez (adanydr@outlook.com)
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Cameron Berg (cam.h.berg@gmail.com)
================================================= """

import collections
import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat


class WalkEnvV0(BaseV0):
    #TODO
    #Reward equation is not working, we need to find a way to improve it.
    DEFAULT_OBS_KEYS = [
        "qpos_without_xy",
        "qvel",
        "com_vel",
        "torso_angle",
        "feet_heights",
        "height",
        'phase_var',
        "feet_rel_positions",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
        'grf',
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        #increase reward per time walking
        "vel_reward": 5.0,
        "done": -20.0,
        "fall_rew": -100.0,
        "cyclic_hip": -10.0,
        "ref_rot": 10.0,
        "joint_angle_rew": 5.0,
        #"mirror_leg" : -10,
        #"not_move_rew" : -5.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(
            self,
            obs_keys: list = DEFAULT_OBS_KEYS,
            weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            max_episode_steps=2000,
            min_height = 0.8,
            max_rot = 0.8,
            hip_period = 100,
            reset_type='init',
            target_x_vel=0.0,
            target_y_vel=0.8,
            target_rot = None,
            **kwargs,
    ):
        self.max_steps = max_episode_steps
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0
        self.help_step = 0
        self.history_size = 200
        self.initial_pos = None
        self.organic_leg = np.empty(self.history_size)
        self.art_leg = np.empty(self.history_size)
        self.grf_sensor_names = ['r_prosth', 'l_foot', 'l_toes']



        super()._setup(
            obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, **kwargs
        )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

        # move heightfield down if not used
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])




    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos_without_xy"] = sim.data.qpos[2:].copy()  ### qpos[2:] original
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        obs_dict["com_vel"] = np.array([self._get_com_velocity().copy()])
        obs_dict["torso_angle"] = np.array([self._get_torso_angle().copy()])
        obs_dict["feet_heights"] = self._get_feet_heights().copy()
        obs_dict["height"] = np.array([self._get_height()]).copy()
        obs_dict["feet_rel_positions"] = self._get_feet_relative_position().copy()
        obs_dict['phase_var'] = np.array([(self.steps/self.hip_period) % 1]).copy()
        obs_dict["muscle_length"] = self.muscle_lengths()
        obs_dict["muscle_velocity"] = self.muscle_velocities()
        obs_dict["muscle_force"] = self.muscle_forces()
        obs_dict['grf'] = self._get_grf().copy()

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        mirror_leg_rew = self._get_mirror_ankle() + self._get_mirror_knee()  ## Modified to mirror movement on the other leg
        ref_rot = self._get_ref_rotation_rew()
        no_move_rew = self._get_staying_in_place()
        joint_angle_rew = self._get_joint_angle_rew(['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l',
                                                     'hip_rotation_r'])
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1) / self.sim.model.na if self.sim.model.na != 0 else 0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('vel_reward', vel_reward),
            ('cyclic_hip', cyclic_hip),
            ('ref_rot', ref_rot),
            #('mirror_leg', mirror_leg_rew),
            ('joint_angle_rew', joint_angle_rew),
            ('act_mag', act_mag),
            ("fall_rew", self._get_fall()),
            #("not_move_rew", no_move_rew),
            # Must keys
            ('sparse', vel_reward),
            ('solved', vel_reward >= 1.0),
            ('done', self._get_done()),
        ))
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if self.np_random.uniform() < 0.5:
            qpos = self.sim.model.key_qpos[2].copy()
            qvel = self.sim.model.key_qvel[2].copy()
        else:
            qpos = self.sim.model.key_qpos[3].copy()
            qvel = self.sim.model.key_qvel[3].copy()
        # randomize qpos coordinates
        # but dont change height or rot state
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        grf = self.obs_dict['grf'].copy()
        if grf[2] > 0:
            grf[2] = 1
        if grf[0] > 0:
            grf[0] = 1
        if grf[1] > 0:
            grf[1] = 1
        self.art_leg[self.help_step] = grf[0]
        self.organic_leg[self.help_step] = grf[1]
        self.help_step += 1
        if (self.help_step>=self.history_size):
            self.help_step=0
        return results

    def reset(self):
        self.steps = 0
        self.help_step = 0
        if self.reset_type == "random":
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == "init":
            qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[1]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel)
        return obs

    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_steps_reward(self):
        return self.steps

    def _get_fall(self):
        height = self._get_height()
        if height < self.min_height:
            return 1
        return 0

    def _get_done(self):
        if self._get_fall():
            return 1
        if self._get_rot_condition():
            return 1
        if self.steps >= self.max_steps:
            return 1
        return 0

    def _get_joint_angle_rew(self, joint_names):
        """
        Get a reward proportional to the specified joint angles.
        """
        mag = 0
        joint_angles = self._get_angle(joint_names)
        mag = np.mean(np.abs(joint_angles))
        return np.exp(-5 * mag)


    def _get_mirror_knee(self):
        phase_var = (self.steps / self.hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)],
                              dtype=np.double)
        angles = self._get_angle(['knee_angle_l', 'osl_knee_angle_r'])
        return np.linalg.norm(des_angles - angles)

    def _get_mirror_ankle(self):
        phase_var = (self.steps / self.hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)],
                              dtype=np.double)
        angles = self._get_angle(['ankle_angle_l', 'osl_ankle_angle_r'])
        return np.linalg.norm(des_angles - angles)


    def _get_mirror_leg_rew(self):
        imitationReward = 0
        if(self.help_step<self.history_size):
            return 0 # Not enough data

        similarity = np.linalg.norm(np.abs(np.sum(self.art_leg) - np.sum(self.organic_leg)))
        #print(similarity)
        if similarity > 10:
            similarity = 10
        imitationReward = np.exp(-similarity)
        self.help_step = 0
        return imitationReward

    def _get_staying_in_place(self):
        """
        Penalise staying close to the initial position of pelvis traslation up to a certain threshold.
        """
        initial_pos = [
            self.initial_pos if self.initial_pos is not None else self.init_qpos[1]
        ][0] ## movement of Y [1], if looking X, is [0]
        res = (self.sim.data.qpos[1] - initial_pos)
        if(res < 0):
            return 1

        diff = np.exp(-res)

        return diff


    def _get_grf(self):
        return np.array([self.sim.data.sensor(sens_name).data[0] for sens_name in self.grf_sensor_names])

    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("osl_foot_assembly")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l][2],
                self.sim.data.body_xpos[foot_id_r][2],
            ]
        )

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("osl_foot_assembly")
        pelvis = self.sim.model.body_name2id("pelvis")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l] - self.sim.data.body_xpos[pelvis],
                self.sim.data.body_xpos[foot_id_r] - self.sim.data.body_xpos[pelvis],
                ]
        )

    def _get_vel_reward(self):
        """
        Gaussian that incentivizes a walking velocity. Going
        over only achieves flat rewards.
        """
        vel = self._get_com_velocity()
        return np.exp(-np.square(self.target_y_vel - vel[1])) + np.exp(
            -np.square(self.target_x_vel - vel[0])
        )

    def _get_cyclic_rew(self):
        """
        Cyclic extension of hip angles is rewarded to incentivize a walking gait.
        """
        phase_var = (self.steps/self.hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)], dtype=np.double)
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return np.linalg.norm(des_angles - angles)

    def _get_ref_rotation_rew(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        target_rot = [
            self.target_rot if self.target_rot is not None else self.init_qpos[3:7]
        ][0]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))

    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id("torso")
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = -self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

    def _get_height(self):
        """
        Get center-of-mass height.
        """
        return self._get_com()[2]

    def _get_knee_condition(self):
        """
        Checks if the agent is on its knees by comparing the distance between the center of mass and the feet.
        """
        feet_heights = self._get_feet_heights()
        torso = self.sim.model.body_name2id("torso")
        com_height = self._get_height()
        if com_height - np.mean(feet_heights) < 0.61:
            return 1
        if np.mean(self.sim.data.body_xpos[torso][2]) - np.mean(feet_heights) < 0.61:
            return 1
        else:
            return 0

    def _get_rot_condition(self):
        """
        MuJoCo specifies the orientation as a quaternion representing the rotation
        from the [1,0,0] vector to the orientation vector. To check if
        a body is facing in the right direction, we can check if the
        quaternion when applied to the vector [1,0,0] as a rotation
        yields a vector with a strong x component.
        """
        # quaternion of root
        quat = self.sim.data.qpos[3:7].copy()
        return [1 if np.abs((quat2mat(quat) @ [1, 0, 0])[0]) > self.max_rot else 0][0]

    def _get_com(self):
        """
        Compute the center of mass of the robot.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com = self.sim.data.xipos
        return np.sum(mass * com, 0) / np.sum(mass)

    def _get_angle(self, names):
        """
        Get the angles of a list of named joints.
        """
        return np.array(
            [
                self.sim.data.qpos[
                    self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]
                ]
                for name in names
            ]
        )
