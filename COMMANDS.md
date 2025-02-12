## Training Commands

```bash
python main.py --train-patients 1 --wandb-run-name "Patient_1"
```

### EndoVis17 Bin Runs

- Run 1 Patient 5
- Run 2 Patient 5,6
- Run 3 Patient 4,5,6
- Run 4 Patient 4,5,6,8
- Run 5 Patient 1-8


rclone copy remote:right-lower-lobe/pa/frames ~/workspace/data/frames --progress --transfers 32
rclone copy remote:right-lower-lobe/pa/masks ~/workspace/data/masks --progress --transfers 32

remote:right-lower-lobe/pa/frames
remote:right-lower-lobe/pa/masks

rclone copy remote:left-lower-lobe/pa/frames ~/workspace/lll_data/frames --progress --transfers 32
rclone copy remote:left-lower-lobe/pa/masks ~/workspace/lll_data/masks --progress --transfers 32