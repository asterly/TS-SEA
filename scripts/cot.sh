# python train_ts_cot_two.py --dataset ISRUC  # --eval  --model_path /workspace/TS-CoT/pretrained_model/two_view/EDF_model.pkl
# python train_ts_cot_two.py --dataset HAR #  --eval  --model_path /workspace/TS-CoT/pretrained_model/two_view/EDF_model.pkl
#python train_ts_cot_two.py --dataset SleepEDF # --eval  --model_path /workspace/TS-CoT/save_dir/test/SleepEDF/TS_CoT/20240321_123957/model.pkl
#python train_ts_cot.py --dataset SleepEDF # --eval  --model_path /workspace/TS-CoT/save_dir/test/SleepEDF/TS_CoT/20240321_151831/model.pkl
#python train_ts_cot_copy.py --dataset EDF --eval   --model_path /workspace/TS-CoT/pretrained_model/two_view/Epi_model.pkl
# python train_ts_cot.py --dataset HAR --eval --model_path /workspace/TS-CoT/save_dir/test/HAR/TS_CoT/20231229_115509/model.pkl
# python train_ts_cot.py --dataset Epi --eval --model_path /workspace/TS-CoT/pretrained_model/best/Epi_three_model.pkl --gpu 1

for run in {1..5}
do 
    python train_ts_cot_two.py --dataset Epi
done