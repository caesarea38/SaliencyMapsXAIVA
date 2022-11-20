ls
cd padl
ls
cd runs
ls
cd default/
ls
cd imbalanced_ssl_training_gcd_mult_runs_v2/
ls
screen
tmux
cd padl/runs/default/
cd imbalanced_ssl_training_gcd_mult_runs_v2/
ls
cd 2022_08_28__03_16_50/logs/
tail training.log 
tail training.log 
tail training.log 
squeue
cd ..
cd ..
cd ..
cd ..
ls
cd ..
ls
pwd
cd ..
cd osr_novel_categories/
ls
cd extracted_features_public_impl/
ls
cd ../
pwd
cd ..
cd padl/runs/default/feature_extraction_gcd_mult_runs_v2/
ls
cd 2022_08_31__17_51_23/
cd logs/
head training.log 
./padl/scripts/interactive.sh 
exit
cd padl/runs
ls
cd default/semi_supervised_k_means_gcd_mult_runs
ls
cd ..
cd semi_supervised_k_means_mult_runs_v2_best
ls
cd semi_supervised_k_means_gcd_mult_runs_v2_best/
ls
cd last
ls
cd logs/
ls
tail - training.log 
tail -f training.log 
pwd
cd ../../..
grep -r "20067" .
grep --exclude "JPEG" -r "20067" .
grep --include=\*.log -r "20067" .
grep --include=\*.out -r "20067" .
cd imbalanced_ssl_training_gcd/2022_08_31__17_46_51/
ls
cd logs/
ls
tail training.log 
less training.log 
squeue
scancel 20067
squeue
sinfo
squeue
sinfo
ls
cd ..
cd ..
cd ..
ls
cd semi_supervised_k_means_gcd_mult_runs_v2_best/
ls
cd last/logs/
ls
tail -f training.log 
cd ..
cd logs/
less training.log 
less training.log 
vim training.log 
cat training.log 
owd
pwd
squeue
ls
cd ..
cd ..
cd ..
cd semi_supervised_k_means_gcd_mult_runs_v2_last/last
ls
cd logs
ls
cat training.log 
NAME="semi_supervised_k_means_gcd_mult_runs_v2_best" ./scripts/prepare.sh python generalized-category-discovery-main/k_means.py --batch_size 128 --num_workers 16 --K 100 --spatial False --semi_sup True --max_kmeans_iter 20 --k_means_init 100 --model_name vit_dino_imb_ssl --dataset_name cifar100 --prop_train_labels 0.5 --warmup_model_exp_id "\(28.08.2022_\|_51.640\)" --use_best_model True --use_ssb_splits False --train_unlabelled_longtailed True
cd ..
cd ..
cd ..
cd ..
cd ..
NAME="semi_supervised_k_means_gcd_mult_runs_v2_best" ./scripts/prepare.sh python generalized-category-discovery-main/k_means.py --batch_size 128 --num_workers 16 --K 100 --spatial False --semi_sup True --max_kmeans_iter 20 --k_means_init 100 --model_name vit_dino_imb_ssl --dataset_name cifar100 --prop_train_labels 0.5 --warmup_model_exp_id "\(28.08.2022_\|_51.640\)" --use_best_model True --use_ssb_splits False --train_unlabelled_longtailed True
NUM_GPU=1 /scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_best/2022_09_02__12_46_07" -w fb10dl05
NUM_GPU=1 ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_best/2022_09_02__12_46_07" -w fb10dl05
NAME="semi_supervised_k_means_gcd_mult_runs_v2_last" ./scripts/prepare.sh python generalized-category-discovery-main/k_means.py --batch_size 128 --num_workers 16 --K 100 --spatial False --semi_sup True --max_kmeans_iter 20 --k_means_init 100 --model_name vit_dino_imb_ssl --dataset_name cifar100 --prop_train_labels 0.5 --warmup_model_exp_id "\(28.08.2022_\|_51.640\)" --use_best_model False --use_ssb_splits False --train_unlabelled_longtailed True
NUM_GPU=1  ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_last/2022_09_02__12_49_31" -w fb10dl08
squeue
sinfo
sinfo
squeue
sinfo
sinfo
ls
cd runs/default/semi_supervised_k_means_gcd_mult_runs_v2_last/last
ls
cat logs/training.log 
cd ..
cd ..
cd ..
cd ..
NAME="semi_supervised_k_means_gcd_mult_runs_v2_last" ./scripts/prepare.sh python generalized-category-discovery-main/k_means.py --batch_size 128 --num_workers 16 --K 100 --spatial False --semi_sup True --max_kmeans_iter 20 --k_means_init 100 --model_name vit_dino_imb_ssl --dataset_name cifar100 --prop_train_labels 0.5 --warmup_model_exp_id "\(28.08.2022_\|_29.801\)" --use_best_model False --use_ssb_splits False --train_unlabelled_longtailed True
NUM_GPU=1 ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_last/2022_09_02__13_00_55"
 -w fb10dl08
NUM_GPU=1 ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_last/2022_09_02__13_00_55" -w fb10dl08
NAME="semi_supervised_k_means_gcd_mult_runs_v2_best" ./scripts/prepare.sh python generalized-category-discovery-main/k_means.py --batch_size 128 --num_workers 16 --K 100 --spatial False --semi_sup True --max_kmeans_iter 20 --k_means_init 100 --model_name vit_dino_imb_ssl --dataset_name cifar100 --prop_train_labels 0.5 --warmup_model_exp_id "\(28.08.2022_\|_29.801\)" --use_best_model True --use_ssb_splits False --train_unlabelled_longtailed True
NUM_GPU=1 ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_best/2022_09_02__13_03_02" -w fb10dl05
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
sinfo
cd runs/default/semi_supervised_k_means_gcd_mult_runs_v2_last/
ls
cd last
ls
cd logs/
ls
tail -f training.log 
sinfo
ls
cd ..
cat experiment.sh 
ls
cd ..
cd ..
cd feature_extraction_gcd_mult_runs_v2/
ls
cd 2022_08_31__17_57_55/
ls
vim experiment.sh 
cd ..
cd ..
cd imbalanced_ssl_training_gcd_mult_runs_v2/
vim
cd padl
NAME="semi_supervised_k_means_gcd_mult_runs_v2_best" ./scripts/prepare.sh python generalized-category-discovery-main/k_means.py --batch_size 128 --num_workers 16 --K 100 --use_best_model True --spatial False --semi_sup True --max_kmeans_iter 20 --k_means_init 100 --model_name vit_dino_imb_ssl --dataset_name cifar100 --prop_train_labels 0.5 --warmup_model_exp_id "\(28.08.2022_\|_18.706\)" --use_best_model True --use_ssb_splits False --train_unlabelled_longtailed True
sinfo
squeue
NUM_GPU=1 ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_best/2022_09_02__11_51_17" -w fb10dl09
cd runs/default/semi_supervised_k_means_gcd_mult_runs_v2_best/2022_09_02__11_51_17/
ls
cd logs/
ls
tail -f training.log 
tail -f slurm.out 
squeue
sinfo
NUM_GPU=1 ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_best/2022_09_02__11_51_17" -w fb10dl05
cd ../../..
cd ..
cd ..
NUM_GPU=1 ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_best/2022_09_02__11_51_17" -w fb10dl05
squeue
skill 20125
scancel 20125
squeue
NAME="semi_supervised_k_means_gcd_mult_runs_v2_last" ./scripts/prepare.sh python generalized-category-discovery-main/k_means.py --batch_size 128 --num_workers 16 --K 100 --use_best_model True --spatial False --semi_sup True --max_kmeans_iter 20 --k_means_init 100 --model_name vit_dino_imb_ssl --dataset_name cifar100 --prop_train_labels 0.5 --warmup_model_exp_id "\(28.08.2022_\|_18.706\)" --use_best_model False --use_ssb_splits False --train_unlabelled_longtailed True
squeue
NUM_GPU=1 ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_last/2022_09_02__12_06_24" -w fb10dl03
./scripts/watch-logs.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_last/2022_09_02__12_06_24"
./scripts/watch-logs.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_last/2022_09_02__12_06_24"
NUM_GPU=1 ./scripts/submit.sh "runs/default/semi_supervised_k_means_gcd_mult_runs_v2_last/2022_09_02__12_06_24" -w fb10dl08
ls
cd runs/default/
ls
cd feature_extraction_gcd_mult_runs_v2/
ls
cd 2022_08_31__17_56_55/
ls
cd logs/
vim training.log 
head training.log 
squeue
sinfo
ls
cd ..
cd ..
ls
cd 2022_08_31__17_56_55/
ls
cd logs/
ls
tail training.log 
ls /osr_novel_categories/extracted_features_public_impl/vit_dino_imb_ssl_cifar100_(28.08.2022_|_51.640)
ls /osr_novel_categories/extracted_features_public_impl/vit_dino_imb_ssl_cifar100_\(28.08.2022_\|_51.640\)
ls ~/osr_novel_categories/extracted_features_public_impl/vit_dino_imb_ssl_cifar100_\(28.08.2022_\|_51.640\)
ls ~/osr_novel_categories/extracted_features_public_impl/
ls
cd ..
cd ..
ls
cd 2022_08_31__17_57_55/
cd logs/
head training.log 
squeue 
salloc -w fb10dl09 --gres gpu:1
squeue 
salloc -w fb10dl05 --gres gpu:1
clear
squeue 
squeue 
sinfo
salloc -w resterampe --gres gpu:1
squeue 
salloc -w fb10dl09 --gres gpu:1
salloc -w resterampe --gres gpu:1
squeue 
salloc -w fb10dl08 --gres gpu:1
squeue 
salloc -w fb10dl05 --gres gpu:1
squeue 
salloc -w fb10dl09 --gres gpu:1
clear
./padl/scripts/interactive.sh 
clear
python generalized-category-discovery-main/extract_features.py --batch_size 128 --num_workers 20 --model_name vit_dino_imb_ssl --dataset cifar100 --warmup_model_dir '/osr_novel_categories/metric_learn_gcd/log/(14.08.2022_|_08.707)/checkpoints/model.pt' --use_best_model False --dataset cifar100
./padl/scripts/interactive.sh 
clear
python generalized-category-discovery-main/extract_features.py --batch_size 128 --num_workers 20 --model_name vit_dino_imb_ssl --dataset cifar100 --warmup_model_dir '/osr_novel_categories/metric_learn_gcd/log/(14.08.2022_|_08.707)/checkpoints/model.pt' --use_best_model False --dataset cifar100
./padl/scripts/interactive.sh 
clear
./padl/scripts/interactive.sh 
exit
sue
squeue 
salloc -w fb10dl03 --gres gpu:1
salloc -w resterampe --gres gpu:1
salloc -w fb10dl05 --gres gpu:1
salloc -w fb10dl08 --gres gpu.1
salloc -w fb10dl08 --gres gpu:1
squeue 
scancel 20166
squeue 
clear
squeue 
salloc -w fb10dl04 --gres gpu:1
salloc -w fb10dl08 --gres gpu:1
clear
squeue 
NAME="imbalanced_ssl_training_gcd" ./scripts/prepare.sh python generalized-category-discovery-main/imbalanced_ssl_training.py --batch_size 128 --num_workers 16 --model_name vit_dino --dataset_name cifar100 --prop_train_labels 0.5 --use_ssb_splits False --grad_from_block 11 --lr 0.1 --rho 0.5 --adaptive True --interpolation 3 --crop_pct 0.875 --gamma 0.1 --momentum 0.9 --weight_decay 5e-5 --epochs 200 --transform imagenet --seed 1 --temperature 1.0 --sup_con_weight 0.35 --n_views 2 --contrast_unlabel_only False --train_unlabelled_longtailed False
cd padl
clear
NAME="imbalanced_ssl_training_gcd" ./scripts/prepare.sh python generalized-category-discovery-main/imbalanced_ssl_training.py --batch_size 128 --num_workers 16 --model_name vit_dino --dataset_name cifar100 --prop_train_labels 0.5 --use_ssb_splits False --grad_from_block 11 --lr 0.1 --rho 0.5 --adaptive True --interpolation 3 --crop_pct 0.875 --gamma 0.1 --momentum 0.9 --weight_decay 5e-5 --epochs 200 --transform imagenet --seed 1 --temperature 1.0 --sup_con_weight 0.35 --n_views 2 --contrast_unlabel_only False --train_unlabelled_longtailed False
NUM_GPU=1 ./scripts/submit.sh "runs/default/imbalanced_ssl_training_gcd/2022_09_03__16_47_14" -w fb10dl09
NUM_GPU=1 ./scripts/submit.sh "runs/default/imbalanced_ssl_training_gcd/2022_09_03__16_47_14" -w fb10dl08
squeue 
scancel 20181
git pull
cd padl
git pull
git pull
exit
squeue 
clear
squeue 
salloc -w fb10dl05 --gres gpu:1
git pull
cd padl
git pull
git add .
git commit -m "updated code base to remove some todos"
git push
clear
salloc -w fb10dl05 --gres gpu:1
cd padl
git pull
git add .
git commit -m "updated code base to remove api keys"
git push
python download_inat18.py 
clear
python download_inat18.py
clear
du -h
df -h
cd ..
clear
ls
cd ..
ls
exit
./scripts/interactive.sh 
./scripts/interactive.sh 
./scripts/interactive.sh 
clear
ls
cd ..
clear
ls
./padl/scripts/interactive.sh 
clear
ls
exit
cd padl
./scripts/interactive.sh 
./scripts/interactive.sh 
./scripts/interactive.sh 
salloc -w fb10dl08 --gres gpu:1
clear
id -G
ls
cd datasets/
ls
cd /home/datasets/
ls
cd vision/
ls
cd ImageNet/
ls
clear
ls
cd ..
clear
ls
mkdir ImageNet/ImageNet-LT
cd /home/padl22t4/padl
clear
ls
cd ..
clear
ls
cd datasets/
ls
mkdir iNaturalist18
cd iNaturalist18/
python download_inat18.py 
salloc -w fb10dl08
clear
ls
cd ..
ls
cd ..
ls
cd padl
ls
clear
ls
NAME="download_inat2018" ./scripts/prepare.sh python /home/padl22t4/datasets/iNaturalist18/download_inat18.py 
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__15_53_55"
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__15_53_55" -w fb10dl08
clear
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__15_53_55" -w fb10dl08
NAME="download_inat2018" ./scripts/prepare.sh python /datasets/iNaturalist18/download_inat18.py 
clear
ls
cd /home/datasets/
ls
cd vision/
ls
cd ImageNet-LT/
ls
clear
cd ..
clear
ls
mkdir iNaturalist18
ls
clear
ls
cd /home/padl22t4/
cd padl
ls
NAME="download_inat2018" ./scripts/prepare.sh python /home/padl22t4/datasets/iNaturalist18/download_inat18.py 
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__16_02_10" -w fb10dl08
clear
NAME="download_inat2018" ./scripts/prepare.sh python /home/padl22t4/datasets/iNaturalist18/download_inat18.py 
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__16_03_33" -w fb10dl08
squeue 
NAME="download_inat2018" ./scripts/prepare.sh python /datasets/iNaturalist18/download_inat18.py 
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__16_05_08" -w fb10dl08
clear
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__16_05_08" -w fb10dl08
NAME="download_inat2018" ./scripts/prepare.sh python /datasets/iNaturalist18/download_inat18.py 
cd /home/datasets/vision/iNaturalist18/
ls
ls
pwd
nano train_val2018.tar.gz
ls
nano train_val2018.tar.gz
ls
clear
ls
ls
nano train_val2018.tar.gz 
rm train_val2018.tar.gz 
clear
ls
cd padl
NAME="download_inat2018" ./scripts/prepare.sh python /datasets/iNaturalist18/download_inat18.py 
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__16_11_37" -w fb10dl08
clear
NAME="download_inat2018" ./scripts/prepar.sh ./datasets/iNaturalist18/download_inat18.sh
NAME="download_inat2018" ./scripts/prepare.sh ./datasets/iNaturalist18/download_inat18.sh
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__16_21_53" -w fb10dl08
./scripts/submit.sh "runs/default/download_inat2018/2022_09_08__16_21_53" -w fb10dl08
clear
rm /enroot_share/padl22t4/padl22t4.sqsh && ./scripts/build-image-enroot.sh 
salloc -w fb10dl08 --gres gpu:21
salloc -w fb10dl08 --gres gpu:1
cd /home/datasets/vision/iNaturalist18/
ls
ls
rm train_val2018.tar.gz 
wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz
ls
rm train_val2018.tar.gz 
clear
pwd
clear
ls
cd ..
ls
cd ImageNet
ls
cd extracted/
ls
cd tgr
cd train/
clear
ls
cd n04548280/
clear
ls
clear
cd ..
ls
cd ..
clear
sl
ls
cd train/
ls
ls | wc -l
nano images.json 
clear
ls
cd n04550184/
ls | wc -l
cd ..
cd n04152593/
ls | wc -l
cd ..
clear
ls
cd ..
clear
ls
squeue 
scancel 20329
squeue 
clear
cd /home/padl22t4/padl
salloc -w fb10dl08 --gres gpu:1
cd padl
./scripts/build-image-enroot.sh 
exit
