# params
data_dir=/l/users/ding.bai/scFoundation/pert/data/
data_name=gse133344_k562gi_oe_pert227_84986_19264_withtotalcount
split=simulation
seed=1
epochs=20
valid_every=1
batch_size=4
accumulation_steps=2
test_batch_size=4
hidden_size=512
train_gene_set_size=0.75
mode=v1
highres=0 # 0
lr=0.01 

model_type=maeautobin
bin_set=autobin_resolution_append #autobin_resolution, bin_2, bin_3, no_bin
finetune_method=frozen # [None,finetune, 'frozen', 'finetune_lr_1'])
singlecell_model_path=../model/models/models.ckpt


workdir=./
cd $workdir

export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

result_dir=/l/users/ding.bai/scFoundation/pert/results_record/pa_adj_weighted_de_loss_4_2_4_times_5-${TIMESTAMP}

mkdir -p ${result_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nnodes=1 --nproc_per_node 4 \
    train_ddp.py \
    --data_dir=${data_dir} \
    --data_name=${data_name} \
    --seed=${seed} \
    --result_dir=${result_dir} \
    --seed=${seed} \
    --epochs=${epochs} \
    --valid_every=${valid_every} \
    --batch_size=${batch_size} \
    --test_batch_size=${test_batch_size} \
    --hidden_size=${hidden_size} \
    --bin_set=${bin_set} \
    --model_type=${model_type} \
    --finetune_method=${finetune_method} \
    --singlecell_model_path=${singlecell_model_path} \
    --mode=${mode} \
    --highres=${highres} \
    --accumulation_steps=${accumulation_steps} \
    --loss='loss_adapt' \
    --ddp_loss_weight=5.0 \
    --lr=${lr} \
    --model_class="GEARS_Model_Pert_Adapter_New" \
    --proj_name="PertAdapt" \
    --exp_name="pa_adj_weighted_de_loss_4_2_4_times_5-${TIMESTAMP}" > ${result_dir}/train.log 2>&1