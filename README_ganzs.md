## Steps:
### 1. Build container
* `bash scripts/docker/build.sh.`
* `bash scripts/docker/launch.sh`
### 2. Download data
* `/workspace/electra/data/create_datasets_from_start.sh`
### 3. Pretraining
* `./run_test.sh`
### 4. Finetuning on Squad
* `bash scripts/finetune_ckpts_on_squad.sh`