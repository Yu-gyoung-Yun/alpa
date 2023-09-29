#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--microbatch) microbatch="$2"
        shift # past argument
        shift # past value
        ;;
        
        -dp|--dp)
        dp="$2"
        shift # past argument
        shift # past value
        ;;
        
        -op|--op)
        op="$2"
        shift # past argument
        shift # past value
        ;;

        -pp|--pp)
        pp="$2"
        shift # past argument
        shift # past value
        ;;

        -bs|--bs)
        bs="$2"
        shift # past argument
        shift # past value
        ;;

        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done


ray stop --force
mkdir -p ./output/${bs}
NCCL_P2P_LEVEL=PIX NCCL_SOCKET_IFNAME=SYS NCCL_SHM_DISABLE=1 NCCL_INCLUDE_DIR="/usr/include/" NCCL_LIB_DIR="/usr/lib/" USE_SYSTEM_NCCL=1 
python3 run_clm_flax_dp.py \
    --output_dir="./output/${bs}" \
    --model_type="gpt2-xl" \
    --config_name="gpt2-xl" \
    --tokenizer_name="gpt2-xl" \
    --dataset_name="oscar" \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --do_train \
    --seq_len 256 \
    --block_size="1024" \
    --per_device_train_batch_size="${bs}" \
    --num_micro_batches $microbatch \
    --operator_parallel $op \
    --pipeline_parallel $pp \
    --dtype="float16" \
    --learning_rate="5e-3" --warmup_steps="1" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="1" > ./output/${bs}/124m_gpt2_layer14_${dp}_${op}_${pp}_out_${microbatch}_seq_len_256.txt