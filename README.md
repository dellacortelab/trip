# TrIP

## Setup

1. Clone the repository
    ```
    git clone https://github.com/bhedelius/TrIP/
    ```

2. Build the TrIP PyTorch NGC container.
    ```
    docker build -t trip .
    ```

3. Start an interactive session in the NGC container
    ```
    docker run -it --gpus all --shm-size=128g --ulimit memlock=-1 --ulimit stack=6710886400 --rm -v ${PWD}/results:/results trip:latest
    ```

## Training

To train, first download the ANI-1x dataset from https://springernature.figshare.com/collections/The_ANI-1ccx_and_ANI-1x_data_sets_coupled-cluster_and_density_functional_theory_properties_for_molecules/4712477 and put it in the /results folder.
```
cd /results
wget https://springernature.figshare.com/ndownloader/files/18112775
cd -
```

Convert to the TrIP format

```
bash scripts/convert_ani1x.sh
```

Then you can begin training

```
bash scripts/train.sh
```

or if you want to train on mutliple GPUs using Distributed Data Parallel:
```
bash scripts/train_multi_gpu.sh
```

## Inference
To run inference on the COMP6 benchmarking set, clone the COMP6 repository into the results folder:
```
cd /results
git clone https://github.com/isayev/COMP6
cd -
```

## Tools
All of these tools share similar arguments. To see the arguments for a certain script, use -h.

### Torsion scan of ephedrine
```
python -m trip.tools.torsion_scan --pdb /results/ephedrine.pdb --atom_nums 13,10,15,18
python -m trip.tools.torsion_scan --pdb /results/ephedrine.pdb --atom_nums 13,10,15,18 --model_file ani1x --label ani
```

### Potential energy surface of H2o
```
python -m trip.tools.pes
python -m trip.tools.pes --model_file ani1x --label ani
```

### Frequency Analysis
Run geometry optimization then frequency analysis. The default molecule is water.
```
python -m trip.tools.freq
```


### Molecular Dynamics
```
python -m trip.tools.md
```

### Energy Sweep
```
python -m trip.tools.energy_sweep
```
