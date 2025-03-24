# Sim-2-real

## PX4 Module
To simplify deploying the networks to a platform, an experimental module in px4 has been created along with a guide. On this page you can read about the training and network conversion, to read more about deploying on an actual drone, check out the [PX4 docs](https://docs.px4.io/main/en/advanced/neural_networks.md)

This example uses the ARL LMF drone platform, by doing a hover flight and measuring a couple of things on your own platform you can customize the sim to train networks specifically for your platform

To train your own networks for use in PX4, start by following the [installation instructions](./2_getting_started.md/#installation). Then you can go into the rl_games folder and run the runner.py script.

```bash
cd aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_task_sim2real_end_to_end
```

If you want to customize the simulator to your own platform or research please check the [Customizing the Simulator](./5_customization.md) section. The relevant files can be found in:

- Robot:
    - Configuration: aerial_gym/config/robot_config/lmf1_config.py
    - Model: resources/robots/lmf1/model.urdf
- Network size, layers and activation function: aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml
- Task:
    - Configuration: aerial_gym/config/task_config/position_setpoint_task_sim2real_end_to_end_config.py
    - Task: aerial_gym/task/position_setpoint_task_sim2real_end_to_end/position_setpoint_task_sim2real_end_to_end.py

In the task you can change the actions, inputs, reward functions, which robot to use etc. Make sure the lmf1 robot is chosen in the task config if you want to use that as a starting point. To train a optimal position setpoint controller for you own platform you should change the robot files. These are the most important changes to make sure the platform in sim is close to the real platform:

- The total weight of the platform in flight, including batteries. This is changed in the urdf file as the mass of the base_link.
- The inertia of the platform, this needs to be calculated using standard methods from measurements of the platform. Added in the urdf file under the inertia of the base link
- The positions of the different motors, how far they are from the middle of the platform. This is changed in the urdf file under each of the base_link_to_..._prop as origin xyz.
- Motor time constants, usually found in the datasheet of the motors. Added in the robot config file as motor_time_constant...
- Motor thrust coefficients, found by doing a hover flight and logging the rpm values needed to hover. The calculation is TODO... Then change the values in the robot config file motor_thrust_constant...

## Conversion
While the Aerial Gym Simulator uses the PyTorch framework, the PX4 module uses TensorFlow Lite Micro (TFLM). Therefore the networks trained in Aerial Gym needs to be converted before they can be added into PX4. Along with the instructions here, a conversion script can be found in the resources/conversion folder.

1. First of all you need to setup the conversion environment. There are some packages here that interferes with the Aerial Gym ones so the recommended way is to exit the conda environment and create a python virtual environment:

    ```bash
    cd resources/conversion
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

1. Then you need to copy in the networks. If you are using the example above you will find the networks in aerial_gym/rl_training/rl_games/runs/{your newest run}/nn/gen_ppo.pth

1. Open the convert.py script and make sure the layer sizes and activation functions match the network you have trained. If you have an example input where you know the correct outputs, you can switch out the sample_input variable to make sure the tflite model produces the correct outputs. Also make sure the correct model name is specified for the network you have copied in.

1. Run the conversion script

    ```bash
    python convert.py
    ```

1. Check that the outputs are correct

1. Run this command with the correct file names: (This is a linux command)

    ```bash
    xxd -i gen_ppo.tflite > gen_ppo.cc
    ```

1. Follow the instructions in the [PX4 docs](https://docs.px4.io/main/en/advanced/tflm.md) to add the network into PX4.