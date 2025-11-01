import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import time

def evaluate_pair_worker(pair_data):
    activation, loss, metrics, device = pair_data
    print(f"Worker evaluating activation '{activation['name']}' with loss '{loss['name']}'")
    try:
        # Create a new evaluator instance for each worker to avoid shared state
        worker_evaluator = Evaluator(metrics=metrics, device=device)
        result = worker_evaluator.evaluate_pair(activation, loss)
        return {
            "activation": activation,
            "loss": loss,
            "metrics": result
        }
    except Exception as e:
        print(f"Error evaluating pair: {str(e)}")
        return {
            "activation": activation,
            "loss": loss,
            "metrics": {"error": str(e)},
            "failed": True
        }

class Evaluator:
    """
    Evaluator class for assessing activation and loss function pairs.
    Handles dataset loading, model training, and performance evaluation.
    """
    
    def __init__(self, metrics=None, device=None, dataset_config=None, save_model_dir: str | None = None):
        """
        Initialize the evaluator with metrics and device.
        
        Args:
            metrics: Dictionary of metric functions to evaluate models
            device: Device to run evaluations on (CPU or GPU)
        """
        # Normalize metrics: accept None, list of names, or dict of callables
        if metrics is None:
            self.metrics = {"accuracy": self._calculate_accuracy}
        elif isinstance(metrics, dict):
            self.metrics = metrics
        elif isinstance(metrics, (list, tuple)):
            mapped = {}
            for m in metrics:
                key = str(m).lower()
                if key in ("accuracy",):
                    mapped["accuracy"] = self._calculate_accuracy
                elif key in ("f1", "f1_score"):
                    mapped["f1_score"] = self._calculate_f1_score
                # 'convergence' is computed separately, skip mapping here
            if not mapped:
                mapped = {"accuracy": self._calculate_accuracy}
            self.metrics = mapped
        else:
            self.metrics = {"accuracy": self._calculate_accuracy}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_cache = {}  # Cache for datasets to avoid reloading
        self.dataset_config = dataset_config
        self.save_model_dir = save_model_dir
        
    def evaluate_population(self, activations: List[Dict[str, Any]], 
                         losses: List[Dict[str, Any]], 
                         parallel: bool = True, 
                         max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Evaluate all pairs of activations and losses, with optional parallel processing.
        
        Args:
            activations: List of activation function configurations
            losses: List of loss function configurations
            parallel: Whether to use parallel processing (default: True)
            max_workers: Maximum number of parallel workers (default: CPU count)
            
        Returns:
            List of evaluation results with scores
        """
        # Import necessary modules for parallel processing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        # Create all pairs to evaluate
        pairs = [(activation, loss, self.metrics, self.device) for activation in activations for loss in losses]
        results = []
        
        # Determine number of workers
        if max_workers is None:
            max_workers = 2
        
        if parallel and len(pairs) > 1:
            print(f"Starting parallel evaluation with {max_workers} workers")
            
            # Use ProcessPoolExecutor for parallel evaluation
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_pair = {executor.submit(evaluate_pair_worker, pair): pair for pair in pairs}
                
                # Collect results as they complete
                for future in as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    try:
                        result = future.result()
                        results.append(result)
                        print(f"Completed evaluation of activation '{pair[0]['name']}' with loss '{pair[1]['name']}'")
                    except Exception as e:
                        print(f"Exception occurred: {str(e)}")
                        results.append({
                            "activation": pair[0],
                            "loss": pair[1],
                            "metrics": {"error": str(e)},
                            "failed": True
                        })
        else:
            # Sequential evaluation
            for activation, loss, _, _ in pairs:
                print(f"Evaluating activation '{activation['name']}' with loss '{loss['name']}'")
                result = self.evaluate_pair(activation, loss)
                fitness = {
                    "accuracy": result.get("accuracy"),
                    "f1_score": result.get("f1_score"),
                    "convergence": result.get("convergence"),
                    "score": result.get("score"),
                    "training_time": result.get("training_time"),
                }
                results.append({
                    "activation": activation,
                    "loss": loss,
                    "metrics": result,
                    "fitness": fitness,
                })
                
        return results
        
    def evaluate_pair(self, activation: Dict[str, Any], loss: Dict[str, Any], 
                      dataset: str = "mnist", epochs: int = 3) -> Dict[str, float]:
        """
        Evaluate a single activation-loss function pair.
        
        Args:
            activation: Activation function configuration
            loss: Loss function configuration
            dataset: Dataset to use for evaluation ("mnist", "fashion_mnist", "cifar10", or "cifar100")
            epochs: Number of training epochs
            
        Returns:
            Dictionary of evaluation metrics
        """
        start_time = time.time()
        
        # Load data (prefer DataModule if dataset_config provided)
        train_loader, val_loader = self._load_dataset(dataset, parallel=self.device != 'cpu')
        
        # Create and train model
        model = self._create_model(activation)
        model = model.to(self.device)
        
        # Get loss function and normalize loss name
        loss_fn = self._get_loss_function(loss)
        loss_name = loss.get("name", "mse").lower()
        
        # Train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                
                # For testing, ensure consistent batch sizes
                if output.size(0) != target.size(0):
                    # Trim to the smaller size
                    min_size = min(output.size(0), target.size(0))
                    output = output[:min_size]
                    target = target[:min_size]
                
                # Handle output/target shape depending on loss type
                if loss_name in ["mse", "mae", "smooth_l1"]:
                    # Regression-style loss over class scores: use one-hot targets
                    target_one_hot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
                    loss_value = loss_fn(output, target_one_hot)
                elif loss_name in ["cross_entropy"]:
                    # Classification loss expects class indices
                    loss_value = loss_fn(output, target.long())
                elif loss_name in ["binary_crossentropy"]:
                    # Binary cross-entropy expects probabilities for each class; apply sigmoid
                    probs = torch.sigmoid(output)
                    target_one_hot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
                    loss_value = loss_fn(probs, target_one_hot)
                else:
                    # Default fallback: try one-hot
                    target_one_hot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
                    loss_value = loss_fn(output, target_one_hot)
                loss_value.backward()
                optimizer.step()
                
                epoch_loss += loss_value.item()

                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss_value.item():.4f}")
            train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluate the model
        model.eval()
        results = {}
        
        with torch.no_grad():
            for name, metric_fn in self.metrics.items():
                metric_value = metric_fn(model, val_loader)
                results[name] = metric_value

        # Ensure required metrics exist
        if "accuracy" not in results:
            results["accuracy"] = self._calculate_accuracy(model, val_loader)
        if "f1_score" not in results and "f1" in self.__dict__.get("metrics", {}):
            # If user provided f1 metric under different key, respect it
            pass
        elif "f1_score" not in results:
            try:
                results["f1_score"] = self._calculate_f1_score(model, val_loader)
            except Exception:
                results["f1_score"] = 0.0
        
        # Add training time
        results["training_time"] = time.time() - start_time

        # Convergence speed evaluation (normalized)
        if len(train_losses) > 1:
            # How quickly the losses decreased (from 0 to 1, where 1 is better)
            initial_loss = train_losses[0]
            final_loss = train_losses[-1]
            convergence_rate = min(1.0, max(0.0, 1.0 - (final_loss / initial_loss)))
        else:
            convergence_rate = 0.5  # Default value
        results["convergence"] = convergence_rate

        # Integral score (robust to missing metrics)
        acc = float(results.get("accuracy", 0.0))
        f1 = float(results.get("f1_score", 0.0))
        conv = float(results.get("convergence", 0.0))
        score = 0.4 * acc + 0.3 * f1 + 0.3 * conv
        results["score"] = score
        results["train_losses"] = train_losses

        # Optionally save model artifact
        if self.save_model_dir:
            try:
                os.makedirs(self.save_model_dir, exist_ok=True)
                act_name = activation.get("name", "act")
                loss_name = loss.get("name", "loss")
                model_path = os.path.join(self.save_model_dir, f"model_{act_name}_{loss_name}.pth")
                torch.save(model.state_dict(), model_path)
                results["model_path"] = model_path
            except Exception as e:
                results["model_path_error"] = str(e)

        return results
    
    def _load_dataset(self, dataset_name: str = "mnist", batch_size: int = 64, parallel: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Loads dataset and creates DataLoaders for training and validation.
        
        Args:
            dataset_name: Name of the dataset to load ("mnist", "fashion_mnist", "cifar10", or "cifar100")
            batch_size: Batch size for DataLoaders
            parallel: Whether running in parallel
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if not parallel and dataset_name in self.dataset_cache:
            return self.dataset_cache[dataset_name]
        
        # If dataset_config provided, use DataModule
        if self.dataset_config is not None:
            try:
                from utils.data_module import DataModule
                dm = DataModule(self.dataset_config)
                return dm.build_loaders()
            except Exception as e:
                print(f"DataModule failed, falling back to default loader: {e}")

        # Define transformations based on dataset
        if dataset_name in ["mnist", "fashion_mnist"]:
            # Grayscale datasets
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            # RGB datasets (CIFAR)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        # Load the appropriate dataset
        if dataset_name == "mnist":
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        elif dataset_name == "fashion_mnist":
            train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        elif dataset_name == "cifar10":
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        elif dataset_name == "cifar100":
            train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Splitting into training and validation sets
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Creating DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Caching for reuse
        self.dataset_cache[dataset_name] = (train_loader, val_loader)
        
        return train_loader, val_loader
        
    def _load_mnist_data(self, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """Legacy method for MNIST data loading, redirects to _load_dataset."""
        return self._load_dataset("mnist", batch_size)
    
    def _create_model(self, activation_config: Dict[str, Any]) -> torch.nn.Module:
        """
        Create a simple neural network model with the specified activation function.
        
        Args:
            activation_config: Activation function configuration
            
        Returns:
            PyTorch neural network model
        """
        from modules.evoactiv_interface import get_activation_function
        
        # Get activation function
        activation_fn = get_activation_function(activation_config)
        
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self, activation):
                super(SimpleModel, self).__init__()
                self.flatten = torch.nn.Flatten()
                self.fc1 = torch.nn.Linear(784, 128)
                self.activation = activation
                self.fc2 = torch.nn.Linear(128, 10)
                
            def forward(self, x):
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.activation(x)
                x = self.fc2(x)
                return x
                
        return SimpleModel(activation_fn)
    
    def _get_loss_function(self, loss_config: Dict[str, Any]) -> torch.nn.Module:
        """
        Get the loss function based on the configuration.
        
        Args:
            loss_config: Loss function configuration
            
        Returns:
            PyTorch loss function
        """
        from modules.evoloss_interface import get_loss_function
        
        return get_loss_function(loss_config)
    
    def _calculate_accuracy(self, model: torch.nn.Module, data_loader: DataLoader) -> float:
        """
        Calculate accuracy of the model on the given data.
        
        Args:
            model: PyTorch model to evaluate
            data_loader: DataLoader with validation data
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                
                # Handle output shape for loss function compatibility
                if outputs.shape[-1] != target.shape[-1] and len(target.shape) == 1:
                    # Convert target to one-hot encoding if needed
                    num_classes = outputs.shape[-1]
                    target_one_hot = torch.zeros(target.size(0), num_classes, device=self.device)
                    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                    target = target_one_hot
                
                _, predicted = torch.max(outputs.data, 1)
                if len(target.shape) > 1 and target.shape[1] > 1:
                    # If target is one-hot encoded, convert back to class indices
                    target_indices = torch.argmax(target, dim=1)
                else:
                    target_indices = target
                    
                total += target_indices.size(0)
                correct += (predicted == target_indices).sum().item()
                
        return correct / total

    def _calculate_f1_score(self, model: torch.nn.Module, data_loader: DataLoader) -> float:
        """
        Calculate f1_score of the model on the given data.
        
        Args:
            model: PyTorch model to evaluate
            data_loader: DataLoader with validation data
            
        Returns:
            F1 score as a float between 0 and 1
        """
        from sklearn.metrics import f1_score
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
        return f1_score(all_targets, all_preds, average="weighted")