""" =================================================
Author  :: Adan Dominguez (adanydr@outlook.com)
================================================= """

import deprl
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from experiments import experimentation
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
        muscleData = muscle
        plt.plot(muscleData, label=muscleName, color=colorValue)
        plt.xlabel("Timesteps")
        plt.ylabel("Torque")
        plotName = "muscle " + muscleName + " force standing"
        plt.title(plotName)
        plt.legend()
        plt.xlim(nInit, nEnd)
        plt.show()


def mujocoToDegrees(valuesList):
    degreesList = []
    for i in valuesList:
        degreesList.append(i/0.015)
    return degreesList

def plotJoint(joint, jointName, colorValue, nSteps, limInf=45, limSup=45):
    # colorValue = blue, yellow, red, green
    nInit = 100
    nEnd = nInit + nSteps
    if len(joint) < nEnd:
        print("Not enough Data, please repeat the experiment")
        return True
    else:
        #jointData = mujocoToDegrees(joint)
        plt.plot(joint, label=jointName, color=colorValue)
        plt.xlabel("Timesteps")
        plt.ylabel("Angle")
        plotName = jointName + " angle standing"
        plt.title(plotName)
        plt.legend()
        plt.xlim(nInit,nEnd)
        plt.ylim(limInf, limSup)
        plt.show()


def oneRun(env, visual, plotFlag, randAction, policy, T):
    if plotFlag:
        gastroc_r = []
        vast_lat_r = []
        bflh_r = []
        gastroc_l = []
        vast_lat_l = []
        bflh_l = []
        r_hip_flexion = []
        r_plantar_flexion = []
        r_knee_flexion = []
        l_hip_flexion = []
        l_plantar_flexion = []
        l_knee_flexion = []

    obs, *_ = env.reset()
    exp = experimentation(env)
    comArray = []
    velocityArray = []


    for ep in range(T):
        if randAction:
            action = env.action_space.sample()
            action = action * 0
        else:
            action = policy(obs)
            comArray.append(env.get_com_public())
            velocityArray.append(env.get_velocity_public()[1])
        if visual:
            env.mj_render()
        if plotFlag:
            muscles = env.sim.data.actuator_force.tolist()
            position = env.sim.data.qpos.tolist()
            gastroc_r.append(muscles[13])
            vast_lat_r.append(muscles[38])
            bflh_r.append(muscles[6])
            gastroc_l.append(muscles[49])
            #vast_lat_l.append(muscles[69])
            bflh_l.append(muscles[46])
            r_hip_flexion.append(position[7])
            r_knee_flexion.append(position[10])
            r_plantar_flexion.append(position[12])
            l_hip_flexion.append(position[15])
            l_knee_flexion.append(position[18])
            # l_plantar_flexion.append(position[12]) # When Ankle movement exist, with Active TP
            # motor_action.append(action[])
        next_state, reward, done, info, extra = env.step(action)
        obs = next_state
    velocityDeviation, velError = exp.getVelocityDeviation(velocityArray)
    displacements, sway_path, comerror = exp.swayCalculation(comArray)
    # exp.plot_graph(displacements, "CoM Deviation Error (%)", "step", "CoM Deviation", comerror)

    exp.plot_graph(velocityDeviation, "velocityDeviation", "step", "velocityDeviation", velError)
    tot_distance = exp.get_walking_distance(env)
    print("Total distance: ", tot_distance, " metres travelled at a mean velocity of: ", np.mean(velocityArray[-200:]), " m/s.")
    print("Target Velocity: ", exp.target_velocity, " m/s.")
    print("Reward: ", reward)
    env.close()
    if plotFlag:
        muscles = [gastroc_r, vast_lat_r, bflh_r, gastroc_l, vast_lat_l, bflh_l]
        joints = [r_hip_flexion, r_knee_flexion, r_plantar_flexion, l_hip_flexion, l_knee_flexion, l_plantar_flexion]
        return muscles, joints


def multipleRun(env, visual, plotFlag, randAction, policy, totEpisodes):
    if plotFlag:
        gastroc_r = []
        soleus_r = []
        r_hip_flexion = []
        r_plantar_flexion = []
        r_knee_flexion = []
        l_hip_flexion = []
        l_plantar_flexion = []
        l_knee_flexion = []

    obs = env.reset()

    for ep in range(totEpisodes):
        print(f"Episode: {ep + 1} of {totEpisodes}")
        obs, *_ = env.reset()
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
                r_hip_flexion.append(position[7])
                r_knee_flexion.append(position[10])
                r_plantar_flexion.append(position[12])
                l_hip_flexion.append(position[15])
                l_knee_flexion.append(position[18])
                # l_plantar_flexion.append(position[12]) # When Ankle movement exist, with Active TP
                # motor_action.append(action[])
            next_state, reward, done, info, extra = env.step(action)
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
    env = gym.make(env_string, reset_type="random")

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
            jointTable = joints
            for i in range(0,5):
                jointTable[i] = mujocoToDegrees(joints[i])

            #jointTable = [np.transpose(joints[0]), np.transpose(joints[1]), np.transpose(joints[2]), np.transpose(joints[3]), np.transpose(joints[4]), np.transpose(joints[5])]
            #jointTable = [joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]]

            DF = pd.DataFrame(data = jointTable, index = ["right hip flexion", "right knee flexion", "right ankle flexion", "left hip flexion", "left knee flexion", "left ankle flexion"])
            DF = DF.transpose()
            DF.to_csv("WalkingChallenge\helper\joints.csv")

            failed = plotJoint(jointTable[2], "plantar flexion", "blue", samples)
            plotMuscle(muscles[0], "gastrocnemous medial right", "blue", samples)
            plotMuscle(muscles[1], "vastus lateralis right", "red", samples)
            plotMuscle(muscles[2], "bicep femoral long head right", "green", samples)
            plotMuscle(muscles[3], "gastrocnemous medial left", "blue", samples)
            plotMuscle(muscles[4], "vastus lateralis left", "red", samples)
            plotMuscle(muscles[5], "bicep femoral long head left", "green", samples)
            plotJoint(jointTable[0], "right hip flexion", "blue", samples, limInf=-15, limSup=125)
            plotJoint(jointTable[1], "right knee flexion", "blue", samples, limInf=-5, limSup=150)
            plotJoint(jointTable[2], "right ankle flexion", "blue", samples, limInf=-55, limSup=35)
            plotJoint(jointTable[3], "left hip flexion", "blue", samples, limInf=-15, limSup=125)
            plotJoint(jointTable[4], "left knee flexion", "blue", samples, limInf=-5, limSup=150)

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

    healthy_foldername = "WalkingChallenge\myoLegWalk_20230514\\myoleg\\"
    amp_foldername = "baselines_DEPRL\myo_amputation_1\\"

    env_hand = 'HandReach-v1'
    env_walk = "myoLegWalk-v0"
    # 80 actions and muscles, 34 DoF
    env_rough = "myoLegRoughTerrainWalk-v0"
    env_hilly = 'myoLegHillyTerrainWalk-v0'
    env_stairs = 'myoLegStairTerrainWalk-v0'
    env_chase = "myoChallengeChaseTagP1-v1"
    env_challenge = "myoChallengeChaseTagP2-v0"
    #### Artificial Limb walk
    env_amp_1DoF = 'myoAmp1DoFWalk-v0' # OSLv2 transtibial prosthesis 70 muscles + 1 actuator, 37 DoF, 0 - root, 1:6 tranlation, 7:37 - joints
    env_amp_stand = 'myoAmp1DoFStanding-v0'  ### Working
    env_amp_rough = "myoOSLRoughTerrainWalk-v0"
    env_amp_hilly = 'myoOSLHillyTerrainWalk-v0'

    # OSLv2 Transfemoral movement
    env_oslv2 = 'myoLegWalk_OSL-v2' # oslv2 transfemodal prosthesis model


    env_osl_separated = "OSLChallenge-v0"


    ### Run Track 2024
    env_challenge24 = "myoChallengeRunTrackP1-v0"


    ################################
    ######Selection Begins##########
    ################################

    env_string = env_amp_stand

    gymnasiumFlag = False
    verifyModel = False  # flag to analyse model characteristics, no simulation performed
    namesFlag = True # To print names of muscles and DoF
    visual = True # Visual mujoco representation

    # Action to be performed and plot of the muscles and joints

    randAction = False  # Just for testing random movements. if true, loads a Checkpoint
    plotFlag = False  # Enable if we want plots of muscles and joint movements
    sarcFlag = False  # Sarcopenia on the model enabled or not

    # Behaviour of the simulation #### only one movement for a long time testFlag = True
    # totEpisodes movements: testFlag = False
    testFlag = False  # True run once the time specified in timeRunning, False goes for totEpisodes number, resets every time the model fails.
    samples = 300  # how many samples do we want to get from the plots, if plotFlag is active
    totEpisodes = 5
    timeRunning = 1000  # How many seconds simulation run if using testFlag = True

    if gymnasiumFlag:
        import gymnasium as gym
    else:
        from myosuite.utils import gym

    if env_string == 'myoAmpWalk-v0' or env_string == 'myoChallengeAmputeeWalk-v0':
        foldername = amp_foldername
    elif env_string == 'myoAmp1DoFWalk-v0' or env_string == "myoOSLRoughTerrainWalk-v0" or env_string =='myoOSLHillyTerrainWalk-v0':
        foldername = "WalkingChallenge\myoOSLv2_TT_Hilly\\"
    elif env_string == 'myoAmp1DoFStanding-v0':
        foldername = "WalkingChallenge\myoOSLv2_TT_Standing\\"
    elif env_string == "myoAmpPassiveWalk-v0":
        foldername = "WalkingChallenge\myoAmp_passive_walking\\"
    elif env_string == "myoAmpPassiveStand-v0":
        foldername = "WalkingChallenge\myoAmp_Passive_Stand\\"
    elif env_string == "myoOSLWalk-v0":
        foldername = "WalkingChallenge\myoOSL_Walking\\"
    elif env_string == "myoLegWalk_OSL-v2":
        foldername = "WalkingChallenge\myoOSLv2_Walking\\"
    elif env_string == "myoChallengeRunTrackP1-v0":
        foldername = "WalkingChallenge\Challenge_OSL\\"
    else:
        foldername = healthy_foldername


    if verifyModel:
        env = gym.make(env_string, reset_type="random")
        modelCharacteristics(env, importGymnasium=gymnasiumFlag, getNames=namesFlag)
    else:
        failed = True  ## Loop to get graphs if model falls down, repeating until gathering required samples
        while failed:
            failed = main(env_string=env_string, foldername=foldername, visual=visual, randAction=randAction,
                          plotFlag=plotFlag, sarcFlag=sarcFlag, samples=samples, testFlag=testFlag,
                          tot_episodes=totEpisodes, T=timeRunning)

    print("Process Finished")
