#!/usr/bin/env bash
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

# checkpoints=${checkpoints:-"results/models/base/checkpoints"}
mode=$1
# checkpoints=${checkpoints:-"test_results/models/base/checkpoints"}
if [[ mode == "ganzs" ]]
then
    src=postprocess_pretrained_ckpt_ganzs.py
    checkpoints=test_results/models/base/checkpoints
    electra_model=test_results/models/base/checkpoints/discriminator
else
    src=postprocess_pretrained_ckpt.py
    checkpoints=results/models/test/checkpoints
    electra_model=results/models/test/checkpoints/discriminator
fi

for folder in $checkpoints; do

    ckpts_dir=${folder}
    output_dir=${folder}

    for f in $ckpts_dir/*.index; do
        ckpt=${f%.*}
        echo "==================================== START $ckpt ===================================="
        python postprocess_pretrained_ckpt_ganzs.py --pretrained_checkpoint=$ckpt --output_dir=$output_dir --amp
        bash scripts/run_squad.sh $(source scripts/configs/squad_config.sh && rtx3090_1gpu_amp_local) train_eval;
        echo "====================================  END $ckpt  ====================================";
    done
done
#bash scripts/run_squad.sh $(source scripts/configs/squad_config.sh && dgxa100_8gpu_amp) train_eval
# bash scripts/run_squad.sh results/models/base/checkpoints/discriminator;