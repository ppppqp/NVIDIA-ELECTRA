#!/bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID
train_batch_size_p1=${1:-44}
learning_rate_p1=${2:-"6e-3"}
precision=${3:-"amp"}
num_gpus=${4:-1}
xla=${5:-"xla"}
warmup_steps_p1=${6:-"5"}
train_steps_p1=${7:-500}
save_checkpoint_steps=${8:-500}
resume_training=${9:-"false"}
optimizer=${10:-"lamb"}
accumulate_gradients=${11:-"true"}
gradient_accumulation_steps_p1=${12:-48}
seed=${13:-12439}
job_name=${14:-"electra_lamb_pretraining"}
train_batch_size_p2=${15:-24}
learning_rate_p2=${16:-"4e-3"}
warmup_steps_p2=${17:-"200"}
train_steps_p2=${18:-933}
gradient_accumulation_steps_p2=${19:-144}
ELECTRA_MODEL=${20:-"base"}
DATASET_P1="tfrecord_lower_case_1_seq_len_128_random_seed_12345/books_wiki_en_corpus/train/pretrain_data*" # change this for other datasets
DATA_DIR_P1=${21:-"$DATA_PREP_WORKING_DIR/$DATASET_P1"}
DATASET_P2="tfrecord_lower_case_1_seq_len_512_random_seed_12345/books_wiki_en_corpus/train/pretrain_data*" # change this for other datasets
DATA_DIR_P2=${22:-"$DATA_PREP_WORKING_DIR/$DATASET_P2"}
CODEDIR=${23:-"/workspace/electra"}
init_checkpoint=${24:-"None"}
restore_checkpoint=${restore_checkpoint:-"true"}
RESULTS_DIR=$CODEDIR/test_results


PREFIX=""
TEST_RESULT=$(awk 'BEGIN {print ('1' <= '${num_gpus}')}')
if [ "$TEST_RESULT" == 1 ] ; then
    PREFIX="horovodrun -np $num_gpus "
fi

if [ "$precision" = "amp" ] ; then
   PREC="--amp "
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$xla" = "xla" ] ; then
   PREC="$PREC --xla"
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_p1"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--restore_checkpoint=latest"
fi

if [ "$init_checkpoint" != "None" ] ; then
   CHECKPOINT="--restore_checkpoint=$init_checkpoint"
fi

CMD=" $CODEDIR/test.py"
CMD+=" --model_name=${ELECTRA_MODEL}"
CMD+=" --pretrain_tfrecords=$DATA_DIR_P1"
CMD+=" --model_size=${ELECTRA_MODEL}"
CMD+=" --train_batch_size=$train_batch_size_p1"
CMD+=" --max_seq_length=128 --disc_weight=50.0 --generator_hidden_size=0.3333333 "
CMD+=" --num_train_steps=$train_steps_p1"
CMD+=" --num_warmup_steps=$warmup_steps_p1"
CMD+=" --save_checkpoints_steps=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_p1"
CMD+=" --optimizer=${optimizer} --skip_adaptive --opt_beta_1=0.878 --opt_beta_2=0.974 --lr_decay_power=0.5"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" --log_dir ${RESULTS_DIR} "

CMD="$PREFIX python3 $CMD"
echo "Launch command: $CMD"
$CMD > test_output 2> test_err


echo "finished testing"