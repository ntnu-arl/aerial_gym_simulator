# Sim-2-real

## PX4 Module
To simplify deploying the networks to a platform, an experimental module in px4 has been created along with a guide. On this page you can read about training, network conversion, uploading and building PX4 to test it out for yourself, or build upon it for further research or use-cases.

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

In the task you can change the actions, inputs, reward functions, which robot to use etc. Make sure the lmf1 robot is chosen in the task config if you want to use that as a starting point.

## Optimizing for your platform

To train a optimal position setpoint controller for your own platform you should change the robot files. These are the most important changes to make sure the platform in sim is close to the real platform:

- The total weight of the platform in flight, including batteries. This is changed in the urdf file as the mass of the base_link.
- The inertia of the platform, this needs to be calculated using standard methods from measurements of the platform, or a CAD file. Added in the urdf file under the inertia of the base link
- The positions of the different motors, how far they are from the middle of the platform. This is changed in the urdf file under each of the base_link_to_..._prop as origin xyz.
- Motor time constants, usually found in the datasheet of the motors. Added in the robot config file as motor_time_constant...
- Motor thrust coefficients, found by doing a hover flight and logging the rpm values needed to hover. The calculation is 9.81*platform weight/4 = thrust per motor to keep the drone hovering. Thrust / (rpm^2) = motor thrust coefficient. Then change the values in the robot config file motor_thrust_constant... You can also use rps or rads/s instead of rpm by changing the motor config.

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

1. To use the new network, copy the numbers in the array and just this, the declaration of the network and the definition needs to be as it is in the PX4 module! Paste these over the array elements in src/modules/mc_nn_control/control_net.cpp, then take the size in the bottom of the gen_ppo.cc file and replace the size in the header file; src/modules/mc_nn_control/control_net.hpp.

## Installing PX4

The module is currently not available in the main PX4 repo, because we are waiting for a toolchain upgrade. In the meantime the code can be found in this [fork and branch](https://github.com/SindreMHegre/PX4-Autopilot-public/tree/for_paper). If anything fail with the first three steps, check the [PX4 docs](https://docs.px4.io/v1.15/en/).

1. First clone the branch with its submodules

    ```bash
    git clone --recurse-submodules -b for_paper https://github.com/SindreMHegre/PX4-Autopilot-public.git
    ```

1. Fetch the tags from the main PX4 repo, PX4 will not build without them

    ```bash
    git fetch upstream --tags
    ```

1. Run the toolchain installation script, note that this may break other packages on your computer

    ```bash
    bash ./Tools/setup/ubuntu.sh
    ```

1. Now for the neural part. Add TFLM as a submodule

    ```bash
    git submodule add -b main https://github.com/tensorflow/tflite-micro.git src/lib/tflm/tflite_micro/
    ```

1. Then we need to install the TFLM dependencies. This is automatically done when you build it as a static library, enter the tflite-micro folder and do the following command:

    ```bash
    cd src/lib/tflm/tflite_micro
    ```

    ```bash
    make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m7 microlite
    ```

1. While this is building (it can take a couple of minutes) we can some other changes. The toolchain file in platforms/nuttx/cmake/Toolchain-arm-none-eabi.cmake needs to be edited. In this file you need to add your local path to the PX4-Autopilot repo. This line is marked with a TODO comment.

1. PX4 excludes standard libraries by default, if they are enabled they will break the nuttx build. To get around this we extract some of the standard library header files. This needs to be done after the TFLM make command is finished.

    ```bash
    cd src/lib/tflm
    cp -r tflite_micro/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/arm-none-eabi/include/c++/13.2.1/ include
    rm include/13.2.1/arm-none-eabi/bits/ctype_base.h
    cp ../../modules/mc_nn_control/setup/ctype_base.h include/13.2.1/arm-none-eabi/bits/
    cd ../../..
    ```

1. (Optional) If you want to include the neural network controller module onto a new board, add:

    ```bash
    CONFIG_MODULES_MC_NN_CONTROL=y
    ```
    to your .px4board file. There are three pre-made board config files where other modules are removed to make sure the entire executable fits in the flash memory of the boards. These are: px4_sitl_neural, px4_fmu-v6c_neural and mro_pixracerpro_neural

1. Now everything should be set up and you can build it using the standard make commands

    ```bash
    make px4_sitl_neural
    ```

Warning! When switching to the neural controller on an actual drone there has been a bug some times that the network produces NAN as motor outputs if it does not receive trajectory setpoints. It is advised to change the mc_nn_testing module to setpoints that you desire and start that module first.