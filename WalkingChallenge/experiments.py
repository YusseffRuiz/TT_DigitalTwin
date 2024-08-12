import numpy as np
import matplotlib.pyplot as plt

class experimentation():
    def __init__(self, env):
        self.env = env
        self.initial_com = self.env.get_com_public()
        self.initial_velocity = self.env.get_velocity_public()
        self.target_velocity = self.env.target_y_vel
        self.initial_position = self.env.get_pos_public()

    def swayCalculation(self, CoM_array):
        """

        :param CoM_array: Array of CoM positions across all steps.
        :return:
        displacements: array deviations from the initial CoM position.
        sway_path: Total distance traveled by the CoM.
        """
        displacements = np.linalg.norm(np.array(CoM_array) - self.initial_com, axis=1)

        sway_path = np.sum(np.linalg.norm(np.diff(CoM_array, axis=0), axis=1))

        error = (np.std(displacements) / np.linalg.norm(self.initial_com)) * 100

        return displacements, sway_path, error

    def getVelocityDeviation(self, velocity_array):
        """
        :param velocity_array: Array of velocity positions across all steps.
        :return:
        velocityDeviations: array deviations from the initial velocity.
        """

        velocity_deviation = np.abs(self.target_velocity - np.array(velocity_array))
        error_1 = (np.abs(np.mean(np.array(velocity_array[-200:])) - self.target_velocity))/self.target_velocity
        error = error_1 * 100
        return velocity_deviation, error

    def get_walking_distance(self, env):
        """
                Calculate the total walking distance without falling.

                Returns:
                walking_distance (float): Total distance walked without falling.
        """
        total_distance = (self.initial_position - env.get_pos_public())

        return total_distance

    def get_step_size_and_cadence(self, env):

        #TODO modify everything, this is just an example
        """
        Calculate step size and cadence during walking.

        Returns:
        step_size (float): Average step size.
        cadence (float): Step cadence.
        """
        step_sizes = []
        step_times = []
        hip_flexion_l = 0
        hip_flexion_r = 0
        previous_step_time = 0

        # Detecting step event
        if hip_flexion_l > 0 and previous_step_time is not None:
            step_time = self.env.sim.data.time - previous_step_time
            step_times.append(step_time)
            step_size = np.abs(hip_flexion_l - hip_flexion_r)
            step_sizes.append(step_size)
            previous_step_time = self.env.sim.data.time
        elif hip_flexion_l > 0:
            previous_step_time = self.env.sim.data.time

        if step_times:
            average_step_time = np.mean(step_times)
            cadence = 1 / average_step_time
        else:
            cadence = 0

        step_size = np.mean(step_sizes) if step_sizes else 0

        return step_size, cadence


    def plot_graph(self, data, title, xlabel, ylabel, error = None):
        """
                Plot a graph for the given data.

                Parameters:
                data (np.array): Data to plot.
                title (str): Title of the graph.
                xlabel (str): Label for the x-axis.
                ylabel (str): Label for the y-axis.
        """
        plt.figure()
        plt.plot(data)
        if error is not None:
            plt.title(f"{title} (Error: {error:.2f}%)")
        else:
            plt.title(title)
        #plt.ylim(0, 5)  # Set y-axis limits from 0 to 100%
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


