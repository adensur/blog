To reproduce my results:
- Download [PIDRay](https://github.com/lutao2021/PIDray) dataset (.tar.gz files)
- Unpack them all into the same folder
- Convert using the script provided (needed for labeled finetuning only):
```bash
PYTHONPATH=. python pidray_convert.py --input ~/datasets/pidray --output ~/datasets/pidray_converted
```
All my trainings were done on a single 8*h200 node.  
```bash
# CoCo
PYTHONPATH=. python finetune.py --data ~/datasets/pidray_converted/data.yaml --model yolo11n.pt --epochs 10
# {'metrics/precision(B)': np.float64(0.8218055594596306), 'metrics/recall(B)': np.float64(0.6806772855138542), 'metrics/mAP50(B)': np.float64(0.7562309929593699), 'metrics/mAP50-95(B)': np.float64(0.6230860959833671), 'fitness': np.float64(0.6364005856809674)}


# CoCo_SSL_10
PYTHONPATH=. python ssl_pretrain.py --data ~/datasets/pidray/train --epochs 10 --output out/ssl2 --batch-size 2048 --method distillation
PYTHONPATH=. python finetune.py --data ~/datasets/pidray_converted/data.yaml --model out/ssl2/exported_models/exported_last.pt --epochs 10
# {'metrics/precision(B)': np.float64(0.8098889411618697), 'metrics/recall(B)': np.float64(0.6989087131541459), 'metrics/mAP50(B)': np.float64(0.7686294681794825), 'metrics/mAP50-95(B)': np.float64(0.6325666992073254), 'fitness': np.float64(0.6461729761045412)}

# CoCo_SSL_100
PYTHONPATH=. python ssl_pretrain.py --data ~/datasets/pidray/train --epochs 100 --output out/ssl3 --batch-size 2048 --method distillation
PYTHONPATH=. python finetune.py --data ~/datasets/pidray_converted/data.yaml --model out/ssl3/exported_models/exported_last.pt --epochs 10
```