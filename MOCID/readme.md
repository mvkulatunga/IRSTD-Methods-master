#### File Structure

1. [/utils](MOCID/utils):
    1. [data.py](MOCID/utils/data.py) for dataset preparation.
    2. [eval.py](MOCID/utils/eval.py) for model evaluation using VOC post-2010 all-points continuous integration metric, at IoU of 0.5
    <!--  COCO (101-point metric), by contrast, averages Average Precision across 10 IoU thresholds from $0.50$ to $0.95$ at steps of $0.05$ ($\text{mAP}@[.50:.95]$) and samples precision over 101 fixed recall points (np.linspace(0, 1, 101)). -->
    3. [losses.py](MOCID/utils/losses.py) contains the loss functions, yololoss and iouloss, as defined by MOCID
    4. [utils.py](MOCID/utils/utils.py) contains the model training helper utils, such as ModelEMA (adopted since SSTnet, uses a EMA training strategy)

2. [/components](MOCID/components):
    1. [data.py](MOCID/components/components.py) for model backbone, FPN and head components
    2. [eval.py](MOCID/components/dam.py) for Displacement Aware Mamba components


3. Root:
    1. [config.py](MOCID/config.py) Contains the main model config.
    2. [main.py](MOCID/main.py) entry point to train the model.
    3. [model.py](MOCID/model.py) packaged model.
    4. [train.py](MOCID/train.py) train loop.
