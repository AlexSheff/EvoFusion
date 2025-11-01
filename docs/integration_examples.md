# Integration Examples

This guide shows how to integrate EvoFusion outputs in your pipelines.

## Use Best Activation/Loss in PyTorch
```python
from modules.evoactiv_interface import get_activation_function
from modules.evoloss_interface import get_loss_function

best_activation_config = {"name": "relu", "params": {}}
best_loss_config = {"name": "cross_entropy", "params": {}}

activation = get_activation_function(best_activation_config)
loss_fn = get_loss_function(best_loss_config)

# Example model
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.act = activation
        self.fc2 = torch.nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x)
```

## Load Best Model Artifact
```python
import torch
state_dict = torch.load("results/best_models/model_relu_cross_entropy.pth", map_location="cpu")
model = Model()
model.load_state_dict(state_dict)
```

## Read Final Report
```python
import json
with open("results/reports/final_report.json", "r", encoding="utf-8") as f:
    report = json.load(f)
print(report["best_pair"], report["best_fitness"]) 
```

## Custom Dataset via ImageFolder
```python
# Configure DataModule in Evaluator with a custom dataset
from core.evaluator import Evaluator
custom_dataset_cfg = {
    "name": "image_folder",
    "image_folder_root": "./my_images",  # class subfolders
    "batch_size": 32,
}
Eval = Evaluator(dataset_config=custom_dataset_cfg)
```