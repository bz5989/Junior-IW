# Configs
job_name=new_method_csf_ant
seed=2
option_dim=2

# Run command
python3 -u run/train.py --run_group $job_name \
                        --env ant \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 1 \
                        --normalizer_type preset \
                        --eval_plot_axis -50 50 -50 50 \
                        --trans_optimization_epochs 50 \
                        --n_epochs_per_log 500 \
                        --n_epochs_per_eval 500 \
                        --n_epochs_per_save 10000 \
                        --n_epochs 10001 \
                        --sac_max_buffer_size 1000000 \
                        --algo new_method_metra_sf \
                        --is_new_method True \
                        --noise_type parametrization_option2 \
                        --noise_factor 0 \
                        --discrete 0 \
                        --dim_option $option_dim \
                        --eval_goal_metrics 1 \
                        --goal_range 50 \
                        --turn_off_dones 1 \
                        --sample_new_z 1 \
                        --num_negative_z 256 \
                        --log_sum_exp 1 \
                        --infonce_lam 5.0 \
                        --eval_record_video 0