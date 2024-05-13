#!/bin/bash

# Define the dump location as a variable
DUMP_LOCATION="/media/adamliu/my_archived/FYP_RESULTS"

# List of num_local_steps values to iterate through
STEPS_LIST=(5 10 15 20 25)

# SEED=42
SEED=1234

# Loop through each value in STEPS_LIST
for NUM_LOCAL_STEPS in "${STEPS_LIST[@]}"
do
    # Construct the exp_name dynamically based on num_local_steps
    EXP_NAME="test_local_step_${NUM_LOCAL_STEPS}_l3mvn_zeroshot_seed_${SEED}"

    # Construct wandb_args dynamically
    WANDB_ARGS="project=yadongliu_fyp_eval_test_local_step,name=${EXP_NAME}"

    # Construct the command with the current num_local_steps value, dynamic exp_name, wandb_args, and the dump_location
    COMMAND="python l3mvn_zeroshot_redo.py --agent l3mvn_zeroshot --task_config benchmark/nav/objectnav/objectnav_hm3d_with_semantic.yaml --version v2 --split val --use_gtsem 1 --eval 1 -v 0 --num_processes 9 --wandb_args='${WANDB_ARGS}' --seed $SEED --print_images 1 --auto_gpu_config 0 --map_size_cm 4800 --load L3MVN/pretrained_models/llm_model.pt --num_eval_episodes 80 --num_local_steps $NUM_LOCAL_STEPS --exp_name $EXP_NAME --num_training_frames 200000 --dump_location '$DUMP_LOCATION'"

    # Execute the command
    echo "Executing command with num_local_steps = $NUM_LOCAL_STEPS and exp_name = $EXP_NAME"
    eval $COMMAND

    # Optional: Sleep for a certain time to let the system cool down or to prevent rate limiting, etc.
    echo "Waiting for next execution..."
    sleep 10
done

echo "All simulations completed."
