# Copyright (c) 2022 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xiang Wang, xiangking1995@163.com
# Status: Active

from typing import Optional
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache


@dataclass
class Handler:
    epoch_num: int = 0
    batch_size: int = 0
    epoch_step_num: int = 0
    global_step: int = 0

    max_training_step_num: int = 0
    logging_step: int = 100
    save_step: int = 0
    evaluate_during_training_step: int = 0

    do_evaluate_per_epoch_end: bool = True
    do_save_per_epoch_end: bool = False
    do_save_best_module: bool = False
    is_minimize_metric: bool = False

    should_epoch_stop: bool = False
    should_training_stop: bool = False

    best_score: int = 0
    output_dir: str = 'checkpoint'
    save_best_moulde_metric: str = None

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    @property
    @lru_cache(1)
    def training_step_num(self):
        return self.epoch_step_num * self.epoch_num
