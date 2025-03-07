[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# [Aerial Gym Simulator](index.md)

Welcome to the [Aerial Gym Simulator](https://www.github.com/ntnu-arl/aerial_gym_simulator) repository. Please refer to our [documentation](https://ntnu-arl.github.io/aerial_gym_simulator/) for detailed information on how to get started with the simulator, and how to use it for your research.

The Aerial Gym Simulator is a high-fidelity physics-based simulator for training Micro Aerial Vehicle (MAV) platforms such as multirotors to learn to fly and navigate cluttered environments using learning-based methods. The environments are built upon the underlying [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) simulator. We offer aerial robot models for standard planar quadrotor platforms, as well as fully-actuated platforms and multirotors with arbitrary configurations. These configurations are supported with low-level and high-level geometric controllers that reside on the GPU and provide parallelization for the simultaneous control of thousands of multirotors.

This is the *second release* of the simulator and includes a variety of new features and improvements. Task definition and environment configuration allow for fine-grained customization of all the environment entities without having to deal with large monolithic environment files. A custom rendering framework allows obtaining depth, and segmentation images at high speeds and can be used to simulate custom sensors such as LiDARs with varying properties. The simulator is open-source and is released under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).


Aerial Gym Simulator allows you to train state-based control policies in under a minute:

![Aerial Gym Simulator](./docs/gifs/Aerial%20Gym%20Position%20Control.gif)

And train vision-based navigation policies in under an hour:

![RL for Navigation](./docs/gifs/rl_for_navigation_example.gif)

Equipped with GPU-accelerated and customizable ray-casting based LiDAR and Camera sensors with depth and segmentation capabilities:

![Depth Frames 1](./docs/gifs/camera_depth_frames.gif) ![Lidar Depth Frames 1](./docs/gifs/lidar_depth_frames.gif)

![Seg Frames 1](./docs/gifs/camera_seg_frames.gif) ![Lidar Seg Frames 1](./docs/gifs/lidar_seg_frames.gif)


## Features

- **Modular and Extendable Design** allowing users to easily create custom environments, robots, sensors, tasks, and controllers, and changing parameters programmatically on-the-go by modifying the [Simulation Components](https://ntnu-arl.github.io/aerial_gym_simulator/4_simulation_components).
- **Rewritten from the Ground-Up** to offer very high control over each of the simulation components and capability to extensively [customize](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization) the simulator to your needs.
- **High-Fidelity Physics Engine** leveraging [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym/download), which provides a high-fidelity physics engine for simulating multirotor platforms, with the possibility of adding support for custom physics engine backends and rendering pipelines.
- **Parallelized Geometric Controllers** that reside on the GPU and provide parallelization for the [simultaneous control of (hundreds of) thousands of multirotor](https://ntnu-arl.github.io/aerial_gym_simulator/3_robots_and_controllers/#controllers) vehicles.
- **Custom Rendering Framework** (based on [NVIDIA Warp](https://nvidia.github.io/warp/)) used to design [custom sensors](https://ntnu-arl.github.io/aerial_gym_simulator/8_sensors_and_rendering/#warp-sensors) and perform parallelized kernel-based operations.
- **Modular and Extendable** allowing users to easily create [custom environments](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-environments), [robots](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-robots), [sensors](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-sensors), [tasks](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-tasks), and [controllers](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-controllers).
- **RL-based control and navigation policies** of your choice can be added for robot learning tasks. [Includes scripts to get started with training your own robots.](https://ntnu-arl.github.io/aerial_gym_simulator/6_rl_training).


> [!IMPORTANT] 
> Support for [**Isaac Lab**](https://isaac-sim.github.io/IsaacLab/) and [**Isaac Sim**](https://developer.nvidia.com/isaac/sim) is currently under development. We anticipate releasing this feature in the near future.


Please refer to the paper detailing the previous version of our simulator to get insights into the motivation and the design principles involved in creating the Aerial Gym Simulator: [https://arxiv.org/abs/2305.16510](https://arxiv.org/abs/2305.16510) (link will be updated to reflect the newer version soon!).

## Why Aerial Gym Simulator?

The Aerial Gym Simulator is designed to simulate thousands of MAVs simultaneously and comes equipped with both low and high-level controllers that are used on real-world systems. In addition, the new customized ray-casting allows for superfast rendering of the environment for tasks using depth and segmentation from the environment.

The optimized code in this newer version allows training for motor-command policies for robot control in under a minute and vision-based navigation policies in under an hour. Extensive examples are provided to allow users to get started with training their own policies for their custom robots quickly.


## Citing
When referencing the Aerial Gym Simulator in your research, please cite the following paper

```bibtex
@ARTICLE{kulkarni2025@aerialgym,
  author={Kulkarni, Mihir and Rehberg, Welf and Alexis, Kostas},
  journal={IEEE Robotics and Automation Letters}, 
  title={Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Robots;Robot sensing systems;Rendering (computer graphics);Physics;Engines;Navigation;Training;Motors;Planning;Autonomous aerial vehicles;Aerial Systems: Perception and Autonomy;Machine Learning for Robot Control;Reinforcement Learning},
  doi={10.1109/LRA.2025.3548507}}
```

If you use the reinforcement learning policy provided alongside this simulator for navigation tasks, please cite the following paper:

```bibtex
@INPROCEEDINGS{kulkarni2024@dceRL,
  author={Kulkarni, Mihir and Alexis, Kostas},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding}, 
  year={2024},
  volume={},
  number={},
  pages={15781-15788},
  keywords={Image coding;Navigation;Supervised learning;Noise;Robot sensing systems;Encoding;Odometry},
  doi={10.1109/ICRA57147.2024.10610287}}

```

## Quick Links
For your convenience, here are some quick links to the most important sections of the documentation:

- [Installation](https://ntnu-arl.github.io/aerial_gym_simulator/2_getting_started/#installation)
- [Robots and Controllers](https://ntnu-arl.github.io/aerial_gym_simulator/3_robots_and_controllers)
- [Sensors and Rendering Capabilities](https://ntnu-arl.github.io/aerial_gym_simulator/8_sensors_and_rendering)
- [RL Training](https://ntnu-arl.github.io/aerial_gym_simulator/6_rl_training)
- [Simulation Components](https://ntnu-arl.github.io/aerial_gym_simulator/4_simulation_components)
- [Customization](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization)
- [FAQs and Troubleshooting](https://ntnu-arl.github.io/aerial_gym_simulator/7_FAQ_and_troubleshooting)



## Contact

Mihir Kulkarni  &nbsp;&nbsp;&nbsp; [Email](mailto:mihirk284@gmail.com) &nbsp; [GitHub](https://github.com/mihirk284) &nbsp; [LinkedIn](https://www.linkedin.com/in/mihir-kulkarni-6070b6135/) &nbsp; [X (formerly Twitter)](https://twitter.com/mihirk284)

Welf Rehberg &nbsp;&nbsp;&nbsp;&nbsp; [Email](mailto:welf.rehberg@ntnu.no) &nbsp; [GitHub](https://github.com/Zwoelf12) &nbsp; [LinkedIn](https://www.linkedin.com/in/welfrehberg/)

Theodor J. L. Forgaard &nbsp;&nbsp;&nbsp; [Email](mailto:tjforgaa@stud.ntnu.no) &nbsp; [GitHb](https://github.com/tforgaard) &nbsp; [LinkedIn](https://www.linkedin.com/in/theodor-johannes-line-forgaard-665b5311a/)

Kostas Alexis &nbsp;&nbsp;&nbsp;&nbsp; [Email](mailto:konstantinos.alexis@ntnu.no) &nbsp;  [GitHub](https://github.com/kostas-alexis) &nbsp; 
 [LinkedIn](https://www.linkedin.com/in/kostas-alexis-67713918/) &nbsp; [X (formerly Twitter)](https://twitter.com/arlteam)

This work is done at the [Autonomous Robots Lab](https://www.autonomousrobotslab.com), [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no). For more information, visit our [Website](https://www.autonomousrobotslab.com/).


## Acknowledgements
This material was supported by RESNAV (AFOSR Award No. FA8655-21-1-7033) and SPEAR (Horizon Europe Grant Agreement No. 101119774).

This repository utilizes some of the code and helper scripts from [https://github.com/leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) and [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs).



## FAQs and Troubleshooting 

Please refer to our [website](https://ntnu-arl.github.io/aerial_gym_simulator/7_FAQ_and_troubleshooting/) or to the [Issues](https://github.com/ntnu-arl/aerial_gym_simulator/issues) section in the GitHub repository for more information.
