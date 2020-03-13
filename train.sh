srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=train \
echo $2 $$ \
echo $3 $$ \
python -u -W ignore train.py \
--name $3 --input_image_path $2 \
2>&1 | tee tee_logs/$3_tee.log
#srun --mpi=pmi2 -p $1 -n16 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=1 -w SH-IDC1-10-5-34-[131,134] \

