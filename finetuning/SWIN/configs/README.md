# SWIN Training Configuration Files

This directory contains YAML configuration files for SWIN model training. Each config file defines all the parameters needed for a training run.

## Usage

To use a config file, simply update the `CONFIG_FILE` variable in `train.sh`:

```bash
CONFIG_FILE="configs/swin_l_frozen_v3.yml"
```

Then run your training as usual:
```bash
qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -l gpu_memory=48G -m beas -M faridkar@bu.edu train.sh
```

## Available Configurations

### swin_l_frozen_v3.yml
- **Purpose**: Frozen training (only last 2 transformer blocks + classifier)
- **Model**: SWIN-Large
- **Learning Rate**: 0.005
- **Frozen Type**: v3 (last 2 blocks unfrozen)

### swin_l_unfrozen.yml
- **Purpose**: Full fine-tuning (all layers trainable)
- **Model**: SWIN-Large
- **Learning Rate**: 0.0001 (lower for stability)
- **Frozen Type**: none

## Creating New Configurations

To create a new configuration:

1. Copy an existing config file:
   ```bash
   cp configs/swin_l_frozen_v3.yml configs/my_new_config.yml
   ```

2. Edit the parameters you want to change

3. Update `train.sh` to point to your new config

4. Run training

## Configuration Structure

Each YAML file is organized into sections:

- **model**: Model selection and settings
- **data**: Dataset paths and data processing settings
- **training**: Training hyperparameters (learning rate, batch size, epochs, etc.)
- **custom**: Custom parameters (freezing, run naming)
- **wandb**: Weights & Biases logging configuration

## Frozen Types

- **v1**: Freeze all except last linear layer
- **v2**: Freeze all except last transformer block and linear layer
- **v3**: Freeze all except last 2 transformer blocks and linear layer
- **v4**: Freeze all except last 3 transformer blocks and linear layer

## WandB Logging

All configuration parameters are automatically logged to Weights & Biases, making it easy to track and compare different experiments.
