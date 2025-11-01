# EvoFusion: Unified Coevolution of Activations and Losses

## Overview
EvoFusion is a research framework for the co-evolution of neural network activation functions and loss functions. It implements a genetic algorithm approach to discover optimal combinations of activation and loss functions that improve model performance, convergence speed, and generalization.

## Key Features
- **Co-evolutionary Algorithm**: Simultaneous evolution of activation functions and loss functions
- **PyTorch Integration**: Seamless integration with PyTorch for neural network training and evaluation
- **Modular Architecture**: Easily extensible components for custom activation functions and loss functions
- **Multi-Dataset Support**: Supports MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 datasets
- **Parallel Evaluation**: Efficient parallel evaluation of activation-loss pairs using ProcessPoolExecutor
- **Comprehensive Metrics**: Tracks accuracy, F1 score, and convergence speed
- **Joint Population Evolution**: Co-evolution of activation and loss function populations
- **Fitness Evaluation**: Pair assessment based on metrics (accuracy, F1, convergence)
- **Genetic Operations**: Selection, crossover, and mutation
- **Visualization Tools**: Advanced visualization of evolution progress and symbolic functions
- **Logging and Reporting**: Detailed logs and final reports

## Requirements
- Python 3.10
- Dependencies listed in `requirements.txt`
- PyTorch (CPU-only is sufficient for tests):
  - `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu`

## Quickstart
1) Create and activate a virtual environment
```
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
```
2) Install dependencies
```
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
3) Run unit tests
```
python -m unittest discover -s tests -v
```
4) Run an evolution experiment (MNIST)
```
python -m experiments.run_evolution
```

## Project Structure
- **core/**: Core components of the co-evolution system
  - `fusion_engine.py`: Main evolution cycle controller
  - `evaluator.py`: Evaluates activation-loss pairs using neural networks
  - `population_manager.py`: Manages populations of activations and losses
  - `crossover_mutation.py`: Implements genetic operators
- **modules/**: Interface modules for activation and loss functions
- **utils/**: Utility functions for configuration, metrics, and visualization
- **configs/**: Configuration files for experiments
- **experiments/**: Example experiment scripts
- **results/**: Directory for storing experiment results

## Continuous Integration
GitHub Actions workflow runs unit tests on every push/PR (`.github/workflows/ci.yml`).

## Usage
- Configure via `core/fusion_engine.py::FusionConfig` or `experiments/*.py`.
- Results are saved under `results/`:
  - `results/logs/`, `results/best_models/`, `results/reports/final_report.json`.

## Configuration
The system is configured through YAML files in the `configs/` directory:
- `fusion_default.yaml`: Main configuration for the co-evolution process
- `activ_params.yaml`: Parameters for activation functions
- `loss_params.yaml`: Parameters for loss functions

### YAML Configuration Example (Custom Dataset and Evolution Params)
You can configure EvoFusion entirely via YAML for custom datasets and evolution parameters.

```yaml
# configs/example_custom.yaml

fusion:
  generations: 5
  activ_population_size: 3
  loss_population_size: 3
  metrics: ["accuracy", "f1", "convergence"]
  results_dir: "results"

dataset:
  # Built-ins: mnist | fashion_mnist | cifar10 | cifar100
  # Custom: image_folder (class subfolders inside image_folder_root)
  name: "image_folder"
  image_folder_root: "./my_images"   # folder with class subdirectories
  root: "./data"
  batch_size: 32
  val_split: 0.2
  num_workers: 2
  download: false

training:
  epochs: 3
  seed: 42
```

Load and run using this YAML:

```python
from utils.config_loader import load_config
from core.fusion_engine import FusionEngine, FusionConfig
from core.evaluator import Evaluator

cfg = load_config("configs/example_custom.yaml")

# Build FusionConfig from YAML
fusion_cfg = FusionConfig(**cfg["fusion"])

# Evaluator with DataModule config and model artifact saving
evaluator = Evaluator(
    metrics=fusion_cfg.metrics,
    dataset_config=cfg.get("dataset"),
    save_model_dir="results/best_models",
)

engine = FusionEngine(fusion_cfg, evaluator=evaluator)
summary = engine.run()
print("Best:", summary.get("best_pair"))
print("Report:", summary.get("report_path"))
```

## How to Use
- Save your dataset under `./my_images/<class_name>/...` or switch `dataset.name` to `mnist`, `fashion_mnist`, `cifar10`, or `cifar100`.
- Load YAML via `load_config("configs/example_custom.yaml")`.
- Build `FusionConfig` from `cfg["fusion"]`.
- Create `Evaluator(metrics=fusion_cfg.metrics, dataset_config=cfg["dataset"], save_model_dir="results/best_models")`.
- Run `engine = FusionEngine(fusion_cfg, evaluator=evaluator)` and `engine.run()`.

## Data Loading (DataModule)
- Flexible loader via `utils/data_module.py::DataModule` and `DatasetConfig`.
- Supports built-ins (`mnist`, `fashion_mnist`, `cifar10`, `cifar100`) and custom `image_folder` datasets.
- Configure via YAML `dataset` section or Python dict passed to `Evaluator(dataset_config=...)`.

## Reproducibility
- `utils/reproducibility.py` provides:
  - `set_seed(42)` to fix randomness across Python, NumPy, and PyTorch.
  - `get_env_info()` to log dependency versions and system info.
- `experiments/run_evolution.py` sets seed and prints environment info at start.

## Reporting
- Final report: `results/reports/final_report.json` contains:
  - `best_pair`, `best_fitness`, `history_size`, config snapshot, env metadata.
- Evolution progress plot: `results/reports/evolution_progress.png`.
- Model artifacts: saved to `results/best_models/model_<activation>_<loss>.pth`.

## Integration
- Use best activation/loss via `modules.evoactiv_interface.get_activation_function` and `modules.evoloss_interface.get_loss_function`.
- Load saved model weights into your own models.
- See `docs/integration_examples.md` for end-to-end examples.

## Documentation
For more detailed information, see:
- [Architecture Overview](docs/architecture.md)
- [Usage Guide](docs/usage.md)
- [Research Notes](docs/research_notes.md)
- [Technical Task](techtask.md) for additional details

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the [LICENSE](LICENSE) - see the LICENSE file for details.

## Acknowledgements
- PyTorch team for the deep learning framework
- MNIST dataset creators

EvoFusion unifies evolutionary search for activations (EvoActiv) and loss functions (EvoLoss) into a single co-evolution cycle.