# FAQs and Troubleshooting



## Frequently Asked Questions


!!! question "When is the support for Isaac Lab expected?"
    We are working on supporting Isaac Lab in the near future. Please stay tuned for updates on this.

!!! question "How can I use the Isaac Gym Simulator with my custom robot?"
    The Isaac Gym Simulator is designed to be modular and flexible, allowing users to easily integrate their custom robots. You can refer to the [Custom Robot Integration](./5_customization.md/#custom-robots) section of the documentation for detailed instructions on how to integrate your custom robot with the Isaac Gym Simulator.

!!! question "How can I randomize the pose of the sensor on the robot"
    The pose of the sensor can be randomized by enabling the `randomize_placement` flag in the sensor configuration file. This is however only applicable to Warp rendering pipeline and is very slow with the Isaac Gym's native rendering pipeline as it requires the user to loop through each sensor instance. By default the sensor position is randomized at each reset of the environment, however you can randomize at each timestep if you like with little overhead.

!!! question "How can I change the randomize the pose at which the robot spawns"
    This can be set using the `min_init_state` and the `max_init_state` parameters in the robot configuration file. The starting pose of the robot is randomized at each reset of the environment by default. Based on the current structure, the position is a ratio of the environment bounds, and the orientation can be defined by minimum and maximum roll, pitch and yaw values.


!!! question "**Difference between Environment and Task**"
    A lot of different simulator implementations interchange the terminologies. In our case, we view the environment as the components that define the robot, it's physics surroundings, i.e., the assets near the robot, the parameters of the physics engine that detemine how the various entities in the simulation world interact with one another, and how sensors perceive the data via the sensor parameters.
    
    The task on the other hand is an interpretation of the simulation world and the information provided by / collected from it to reach a particular goal that is deired by the user. The same environment can be used to train multiple tasks and the tasks can be changed without changing an environment definition.
    
    For example, an empty environment with a quadrotor can be used to train a position setpoint task, or a trajectory tracking task. An environment with a set of obstacles can be used to train a policy that can either navigate through the obstacles or perch on a specific asset in the environment. The task is the interpretation of the environment data for the RL algorithm to learn the desired behavior.

    To relate it with a familiar environment from the OpenAI Gym suite of tasks, an "environment" in our case could refer to a CartPole world with the dynamics of the CartPole, however a "task" in our case would allow the same cartpole to be controlled to balance the pole upright, or to keep swinging the pole at a given angular rate or to have the endpoint of the pole at a given location in the environment. All of which require different formulation of rewards and observations for the RL algorithm to learn the desired behavior.




## Troubleshooting

!!! danger "My Isaac Gym viewer window does not show anything or is blank"
    This can be because of a discrepancy in the version of your NVIDIA drivers. Please ensure that you have the appropriate NVIDIA drivers on your system as per the Isaac Gym documentation, and that the Isaac Gym examples such as `1080_balls_of_solitude.py` and `joint_monkey.py` are working as expected. Please also make sure that the environment variables `LD_LIBRARY_PATH` and `VK_ICD_FILENAMES` are set correctly.

!!! danger "rgbImage buffer error 999"
    ```bash
    [Error] [carb.gym.plugin] cudaImportExternalMemory failed on rgbImage buffer with error 999
    ```

    This is most likely due to an improper Vulkan configuration. Please refer to the Troubleshooting section of the Isaac Gym documentation and check if the `VK_ICD_FILENAMES` environment variable is set, and that the file exists.

!!! danger "My simulation assets are going through each other without collisions"
    This occurs in case of an improperly set quaternion for the simulation assets. Kindly check that the quaternion is normalized as required by the Isaac Gym Simulator, and that it is in the format `[q_x, q_y, q_z, q_w]`.

!!! danger "My simulation assets just spin very fast at each reset"
    There seem to be `nan` values somewhere in the measurements of your implementation. This can be due to a variety of reasons, such as improper quaternion normalization, or improper sensor measurements. Please check your code and ensure that all measurements are valid and within the expected range.


!!! warning "I see an error that ends with `if len(self._meshes) == 0:`"
    Most likely you have installed urdfpy package via pip. This version has a bug that has since been resolved in the `master` branch of the URDFPY project repository. Please follow the [installation page](./2_getting_started.md/#installation) to install the package from source.





