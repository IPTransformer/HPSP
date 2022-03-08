CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
--master_addr 127.0.1.17 --master_port=20217 \
train_iptr_cihp.py --opt adam --workers 4 --batch-size 16 \
--epochs 150 --loss-type dice \
--lr 1e-3 \
--dataset cihp \
--backbone res50 \
--img-size 512 \
--nclass 15 \
--ninstance 18 \
--rotate 15 \
--checkname r50_e150_512_rotate20_ins01_bs16_18ins
