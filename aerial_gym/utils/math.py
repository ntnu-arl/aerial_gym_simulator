# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

@torch.jit.script
def compute_vee_map(skew_matrix):
    # type: (Tensor) -> Tensor

    # return vee map of skew matrix
    vee_map = torch.stack(
        [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
    return vee_map