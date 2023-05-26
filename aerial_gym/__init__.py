# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

AERIAL_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
AERIAL_GYM_ENVS_DIR = os.path.join(AERIAL_GYM_ROOT_DIR, 'aerial_gym', 'envs')

print("AERIAL_GYM_ROOT_DIR", AERIAL_GYM_ROOT_DIR)