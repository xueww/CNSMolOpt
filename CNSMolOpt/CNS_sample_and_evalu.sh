CUDA_VISIBLE_DEVICES=0 python -W ignore sample_multi_for_specific_context.py \
        --scaffold_smiles_file /public/home/chensn/experimen/exper_2_19/scaffold_dele_5c.smi \
        --protein_file /public/home/chensn/experimen/exper_2_19/pocket_10A.pdb \
        --scaffold_file /public/home/chensn/experimen/exper_2_19/scaffold_good_dele_5c.sdf \
        --task_name exp \
        --data_dir data/examples_exper_2_19 \
        --checkpoint /public/home/chensn/DL/DiffDec-master/models/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754_best_epoch=epoch=512.ckpt \
        --samples_dir samples_exper_2_28 \
        --n_samples 100 \
        --device cuda:0

python -W ignore eval_true_cns.py \
        --sdf_directory /public/home/chensn/DL/DiffDec-master/samples_exper_c_2_26/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754_best_epoch=epoch=512/0 \
        --pdb_file_path /public/home/chensn/DL/DiffDec-master/samples_exper_c_2_26/multi_chensn_diffdec_multi__avarage_calss2_softmax_test100_bs8_date14-02_time09-45-17.215754_best_epoch=epoch=512/0/pock_.pdb \
        --output_directory /public/home/chensn/DL/DiffDec-master/docking_results_C_sas \
        --cns_sas_output_dir /public/home/chensn/DL/DiffDec-master/CNS_SAS_c \
        --docking_results_csv_dir /public/home/chensn/DL/DiffDec-master/docking_results_csv_c_sas \
        --cns_mpo_env cns_mpo \
