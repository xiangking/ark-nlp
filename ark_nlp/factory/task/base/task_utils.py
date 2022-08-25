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


from dataclasses import dataclass
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Union
from typing import Optional


@dataclass
class Handler:
    epoch: Optional[float] = None
    step: int = 0
    global_step: int = 0

    evaluate_per_epoch_end: bool = False
    save_per_epoch_end: bool = False

    should_epoch_stop: bool = False
    should_training_stop: bool = False

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))