
# for run in {1..5}
# do 
#     # python train_ts_cot.py --dataset HAR --gpu 0 --run_desc three_view_1p --data_perc 'perc' --backbone_type TS_SEA
#     # python train_ts_cot.py --dataset Epilepsy --gpu 0 --run_desc three_view_1p --data_perc '1perc' --backbone_type TS_SEA
#     python train_ts_cot.py --dataset ISRUC --gpu 0 --run_desc few_label --data_perc '1perc' --backbone_type TS_SEA --eval --model_path save_dir/test/ISRUC/TS_SEA/20240408_040456/model.pkl
# done  "50p" "75p"


lab_percs=("256")

for lab_perc in "${lab_percs[@]}"
do
    # 
    python train_ts_cot.py --dataset HAR --gpu 0 --run_desc batch_size --backbone_type TS_SEA   --batch_size  $lab_percs
    python train_ts_cot.py --dataset Epilepsy --gpu 0 --run_desc batch_size --backbone_type TS_SEA   --batch_size  $lab_percs
    python train_ts_cot.py --dataset ISRUC --gpu 0 --run_desc batch_size --backbone_type TS_SEA --batch_size  $lab_percs
    # python train_ts_cot.py --dataset ISRUC --gpu 1 --run_desc tri_view_$lab_perc --data_perc $lab_perc"erc"  --backbone_type TS_SEA
    # python train_ts_cot.py --dataset Epilepsy --gpu 1 --run_desc tri_view_$lab_perc --data_perc $lab_perc"erc"  --backbone_type TS_SEA
    # python3 main.py --experiment_description $exp_desc --run_description $run --seed $i --selected_dataset $dataset --training_mode "ft_$lab_perc"   --device $device 
    # python3 main.py --experiment_description $exp_desc --run_description $run --seed $i --selected_dataset $dataset --training_mode "gen_pseudo_labels" 
    # python3 main.py --experiment_description $exp_desc --run_description $run --seed $i --selected_dataset $dataset --training_mode "SupCon" 
    # python3 main.py --experiment_description $exp_desc --run_description $run --seed $i --selected_dataset $dataset --training_mode "train_linear_SupCon_$lab_perc" 
    
done