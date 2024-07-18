# Introduction

This documentation provides information about the Aerial Gym Simulator, a powerful tool for training multirotor platforms to learn to fly and navigate cluttered environments using learning-based methods. The simulator is built upon the underlying NVIDIA Isaac Gym simulator and offers a range of features and capabilities.

The Aerial Gym Simulator provides simplified multirotor models with high-level geometric controllers and motor controllers that reside on the GPU. This allows for parallelization, enabling the simultaneous control of thousands of multirotor vehicles. The simulator supports various robot configurations, including underactuated quadrotors, fully-actuated octarotors, and arbitrary configurations with customizable designs.

One of the key features of the simulator is its ability to simulate environments with or without obstacles. Users can populate environments with randomly generated obstacles from a library of assets, or create custom obstacle configurations. The simulator also supports the integration of various sensors, such as depth cameras, LiDARs, and IMUs, providing realistic sensor data for training and evaluation.

The simulator is highly configurable, allowing users to customize various aspects of the simulation, including physics parameters, robot configurations, sensor configurations, and environment setups. This flexibility enables users to tailor the simulator to their specific research or application needs.

To get started with the Aerial Gym Simulator, refer to the [Getting Started](./2_getting_started.md) section for installation instructions and examples. The [Robots and Sensors](./3_robots_and_controllers.md) section provides detailed information about the supported robot configurations and sensor models. The [Simulation Components](./4_simulation_components.md) section explains the modular structure of the simulator, including registries, simulation parameters, assets, environments, tasks, controllers, and sensors. The [Customization](./5_customization.md) section outlines the process of creating custom robots, environments, tasks, controllers, and physics parameters.

Additionally, the simulator supports integration with popular reinforcement learning frameworks, such as [rl-games](./6_rl_training.md#rl-games), [sample-factory](./6_rl_training.md#sample-factory), and [CleanRL](./6_rl_training.md#cleanrl). Examples and guidelines for training with these frameworks are provided in the [Reinforcement Learning](./6_rl_training.md) section.

Finally, the [FAQ and Troubleshooting](./7_FAQ_and_troubleshooting.md) section addresses common questions and issues that users may encounter while working with the Aerial Gym Simulator.

By leveraging the Aerial Gym Simulator, researchers and developers can efficiently train and evaluate multirotor platforms in realistic, simulated environments, accelerating the development of advanced navigation and control algorithms for autonomous aerial vehicles.