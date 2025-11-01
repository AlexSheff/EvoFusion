# Running EvoFusion

## Installing Dependencies

1. Install PyTorch from the local WHL in the project root:
   ```
   pip install .\torch-2.4.0+cu118-cp310-cp310-win_amd64.whl
   ```
2. Install basic dependencies:
   ```
   pip install -r requirements.txt
   ```

## Launch Example

```bash
python -m experiments.mnist_fusion_test
```

## Configs
- Main: `configs/fusion_default.yaml`
- Activation parameters: `configs/activ_params.yaml`
- Loss parameters: `configs/loss_params.yaml`