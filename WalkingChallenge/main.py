""" =================================================
Author  :: Adan Dominguez (adanydr@outlook.com)
================================================= """

import myosuite
import deprl
import numpy as np
import matplotlib.pyplot as plt
import torch

import cv2
import os


def modelCharacteristics(env, importGymnasium=False, getNames=False):
    # Testing of muscles and actuators
    if not importGymnasium:
        position = env.sim.data.qpos.tolist()
        velocity = env.sim.data.qvel.tolist()
        muscles = env.sim.data.actuator_force.tolist()
        tendons_len = env.sim.data.ten_length.tolist()
        action = env.action_space.sample()

        print("total number of DoF in the myosuite model: ", env.sim.model.nv)
        print("Generalised positions: ", env.sim.data.qpos, "\n number of positions: ", len(env.sim.data.qpos))
        print("Generalised velocities: ", env.sim.data.qvel)
        # print("position: ", position)

        print("total muscles: ", len(muscles))
        print("Muscles: ", muscles)
        print("total Tendons: ", tendons_len)
        print("Number of actions in the model: ", len(action))
        print("Action: ", action)

        if getNames:
            # get the names of every group
            for i in range(env.sim.model.ngeom):
                print('name of geom ', i, ' : ', env.sim.model.geom(i).name)
            for i in range(env.sim.model.njnt):
                print('name of joints ', i, ' : ', env.sim.model.joint(i).name)
            for i in range(len(tendons_len)):
                print('name of tendons: ', i, ' : ', env.sim.model.tendon(i).name)

    else:
        position = env.data.qpos.tolist()
        velocity = env.data.qvel.tolist()
        muscles = env.data.actuator_force.tolist()
        tendons_len = env.data.ten_length.tolist()
        action = env.action_space.sample()

        print("total number of DoF in the model: ", env.model.nv)
        print("position: ", position)

        print("total muscles: ", len(muscles))
        print("Muscles: ", muscles)
        print("total Tendons: ", tendons_len)
        print("Number of actions in the model: ", len(action))
        print("Action: ", action)

        if getNames:
            # get the names of every group
            # for i in range(len(env.model.geom)):
            for i in range(env.model.ntendon):
                print('name of geom ', i, ' : ', env.model.ntendon(i).name)


def plotMuscle(muscle, muscleName, colorValue, nSteps):
    # colorValue = blue, yellow, red, green
    nInit = 100
    nEnd = nInit + nSteps
    if len(muscle) < nEnd:
        print("Not enough Data, please repeat the experiment")
        return True
    else:
        muscleData = muscle[nInit:nEnd]
        plt.plot(muscleData, label=muscleName, color=colorValue)
        plt.xlabel("Timesteps")
        plt.ylabel("Torque")
        plotName = "muscle " + muscleName + " force standing"
        plt.title(plotName)
        plt.legend()
        plt.show()


def plotJoint(joint, jointName, colorValue, nSteps):
    # colorValue = blue, yellow, red, green
    nInit = 100
    nEnd = nInit + nSteps
    if len(joint) < nEnd:
        print("Not enough Data, please repeat the experiment")
        return True
    else:
        jointData = joint[nInit:nEnd]
        plt.plot(jointData, label=jointName, color=colorValue)
        plt.xlabel("Timesteps")
        plt.ylabel("Angle")
        plotName = jointName + " angle standing"
        plt.title(plotName)
        plt.legend()
        plt.show()


def oneRun(env, visual, plotFlag, randAction, policy, T):
    if plotFlag:
        gastroc_r = []
        vast_lat_r = []
        bflh_r = []
        gastroc_l = []
        vast_lat_l = []
        bflh_l = []
        hip_flexion = []
        plantar_flexion = []
        knee_flexion = []

    obs = env.reset()

    for ep in range(T):
        if randAction:
            action = env.action_space.sample()
            action = action * 0
        else:
            action = policy(obs)
        if visual:
            env.mj_render()
        if plotFlag:
            muscles = env.sim.data.actuator_force.tolist()
            position = env.sim.data.qpos.tolist()
            gastroc_r.append(muscles[13])
            vast_lat_r.append(muscles[38])
            bflh_r.append(muscles[6])
            gastroc_l.append(muscles[49])
            vast_lat_l.append(muscles[69])
            bflh_l.append(muscles[46])
            hip_flexion.append(position[7])
            plantar_flexion.append(position[10])
            knee_flexion.append(position[11])
            # motor_action.append(action[])
        next_state, reward, done, info = env.step(action)
        obs = next_state
    print("Reward: ", reward)
    env.close()
    if plotFlag:
        muscles = [gastroc_r, vast_lat_r, bflh_r, gastroc_l, vast_lat_l, bflh_l]
        joints = [hip_flexion, plantar_flexion, knee_flexion]
        return muscles, joints


def multipleRun(env, visual, plotFlag, randAction, policy, totEpisodes):
    if plotFlag:
        gastroc_r = []
        soleus_r = []
        hip_flexion = []
        plantar_flexion = []
        knee_flexion = []

    obs = env.reset()

    for ep in range(totEpisodes):
        print(f"Episode: {ep + 1} of {totEpisodes}")
        obs = env.reset()
        done = False
        while not done:
            if randAction:
                action = env.action_space.sample()
                action = action * 0
            else:
                action = policy(obs)
            if visual:
                env.mj_render()
            if plotFlag:
                muscles = env.sim.data.actuator_force.tolist()
                position = env.sim.data.qpos.tolist()
                gastroc_r.append(muscles[13])
                soleus_r.append(muscles[33])
                hip_flexion.append(position[7])
                plantar_flexion.append(position[10])
                knee_flexion.append(position[11])
                # motor_action.append(action[])
            next_state, reward, done, info = env.step(action)
            obs = next_state
        print("Reward: ", reward)
    env.close()


def main(env_string, foldername, visual, randAction, plotFlag, sarcFlag, samples, testFlag=False, tot_episodes=5,
         T=500):
    # Sarcopedia Flag only replace "myo" with "myoSarc" the weakness on muscles is added automatically
    if sarcFlag:
        env_string = env_string.replace("myo", "myoSarc")
        print("Walking with Sarcopenia Model")
    else:
        print("Walking with healthy Model")

    # Initialise environment
    env = gym.make(env_string, reset_type="init")
    obs = env.reset()
    action = env.action_space.sample()

    if not randAction:
        print(foldername)
        policy = deprl.load(foldername, env)
    else:
        policy = None

    """
    Muscles are the one we need to keep tracking, action space only indicates the torque required to perform movement on the muscles.

    """
    failed = False

    if testFlag:
        if plotFlag:
            muscles, joints = oneRun(env, visual, plotFlag, randAction, policy, T)

            plotMuscle(muscles[0], "gastrocnemous medial right", "blue", samples)
            plotMuscle(muscles[1], "vastus lateralis right", "red", samples)
            plotMuscle(muscles[2], "bicep femoral long head right", "green", samples)
            plotMuscle(muscles[3], "gastrocnemous medial left", "blue", samples)
            plotMuscle(muscles[4], "vastus lateralis left", "red", samples)
            plotMuscle(muscles[5], "bicep femoral long head left", "green", samples)
            plotJoint(joints[0], "hip flexion", "blue", samples)
            plotJoint(joints[1], "knee flexion", "blue", samples)
            failed = plotJoint(joints[2], "plantar flexion", "blue", samples)

        else:
            oneRun(env, visual, plotFlag, randAction, policy, T)
    else:
        multipleRun(env, visual, plotFlag, randAction, policy, tot_episodes)

    if failed:
        return True
    else:
        return False


# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    healthy_foldername = "baselines_DEPRL\myoLegWalk_20230514\myoLeg\\"
    amp_foldername = "baselines_DEPRL\myo_amputation_1\\"

    env_hand = 'HandReach-v1'
    env_walk = "myoLegWalk-v0"
    # 80 actions and muscles, 34 DoF
    env_rough = "myoLegRoughTerrainWalk-v0"
    env_hilly = 'myoLegHillyTerrainWalk-v0'
    env_stairs = 'myoLegStairTerrainWalk-v0'
    env_chase = "myoChallengeChaseTagP1-v1"
    env_challenge = "myoChallengeChaseTagP2-v0"
    #### Artificial Limb walk 71 actions (70 muscles, 1 motor), 20 DoF.
    env_amp_2DoF = 'myoAmpWalk-v0'
    env_amp_1DoF = 'myoAmp1DoFWalk-v0'
    env_amp_challenge = 'myoChallengeAmputeeWalk-v1'
    env_amp_stand = 'myoAmp1DoFStand-v0'

    ################################
    ######Selection Begins##########
    ################################

    env_string = env_amp_1DoF

    gymnasiumFlag = False
    verifyModel = False  # flag to analyse model characteristics, no simulation performed
    namesFlag = True
    visual = True  # Visual mujoco representation

    # Action to be performed and plot of the muscles and joints

    randAction = False  # Just for testing random movements
    plotFlag = False  # Enable if we want plots of muscles and joint movement
    sarcFlag = False  # Sarcopenia on the model enabled or not

    # Behaviour of the simulation #### only one movement for a long time testFlag = True
    # totEpisodes movements: testFlag = False
    testFlag = True  # True run once the time specified in timeRunning, False goes for totEpisodes number, resets every time the model fails.
    samples = 300  # how many samples do we want to get from the plots, if plotFlag is active
    totEpisodes = 5
    timeRunning = 1000  # How many seconds simulation run if using testFlag = True
    #
    if gymnasiumFlag:
        import gymnasium as gym
    else:
        import gym

    if env_string == 'myoAmpWalk-v0' or env_string == 'myoChallengeAmputeeWalk-v0':
        foldername = amp_foldername
    elif env_string == 'myoAmp1DoFWalk-v0':
        foldername = "WalkingChallenge\myoAmp_walking_1\\"
    elif env_string == 'myoAmp1DoFStand-v0':
        foldername = "myoAmp_walking\\"
    else:
        foldername = healthy_foldername

    # 34 DoF 78 actions and muscles
    if verifyModel:
        env = gym.make(env_string)
        modelCharacteristics(env, importGymnasium=gymnasiumFlag, getNames=namesFlag)
    else:
        failed = True  ## Loop to get graphs if model falls down, repeating until gathering required samples
        while failed:
            failed = main(env_string=env_string, foldername=foldername, visual=visual, randAction=randAction,
                          plotFlag=plotFlag, sarcFlag=sarcFlag, samples=samples, testFlag=testFlag,
                          tot_episodes=totEpisodes, T=timeRunning)

    print("Process Finished")
