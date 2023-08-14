#inference
python3 test.py --data_path data/Sketch/paper_version/val --resume checkpoint/checkpoint.pth

#eval
python3 main.py --num_workers 14 --batch_size 4 --device "cuda:0" --eval --resume checkpoint/checkpoint.pth

#train
python3 main.py --num_workers 14 --batch_size 4 --device "cuda:0" --resume detr-r50-e632da11.pth
