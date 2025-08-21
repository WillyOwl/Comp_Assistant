"""
Hyperparameter optimization for composition analysis models.

This module implements automated hyperparameter tuning using Optuna and other
optimization frameworks to find optimal training configurations.
"""

import optuna
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
from .trainer import CompositionTrainer
from .dataset_loader import create_data_loaders

logger = logging.getLogger(__name__)

class CompositionHyperparameterOptimizer:
    """
    Hyperparameter optimizer for composition analysis models.
    """

    def __init__(self, base_config: Dict[str, Any],
                 optimization_config: Dict[str, Any],
                 n_trials: int = 100, study_name: Optional[str] = None):
        
        """
        Initialize hyperparameter optimizer.
        
        Args:
            base_config: Base training configuration
            optimization_config: Hyperparameter search space configuration
            n_trials: Number of optimization trials
            study_name: Name for the optimization study
        
        """

        self.base_config = base_config
        self.optimization_config = optimization_config
        self.n_trials = n_trials
        self.study_name = study_name or "composition_optimization"

        # Create output directory for optimization results

        self.output_dir = Path(base_config['output_dir']) / 'hyperparameter_optimization'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize study
        self.study = None
        self._init_study()

    def _init_study(self):
        """Initialize Optuna study with appropriate sampler and pruner."""

        # Use TPE sampler for efficient hyperparameter search
        sampler = optuna.samplers.TPESampler(
            n_startup_trials = 10,
            n_ei_candidates = 24,
            seed = 42
        )

        # Use median pruner to stop unpromising trials early
        pruner = optuna.pruners.MedianPruner(
            n_min_trials = 5,
            n_warmup_steps = 10,
            interval_steps = 1
        )

        # Create or load study
        storage_url = f"sqlite:///{self.output_dir}/{self.study_name}.db"
        self.study = optuna.create_study(
            study_name = self.study_name,
            storage = storage_url,
            load_if_exists = True,
            direction = 'maximize',  # Maximize validation score
            sampler = sampler,
            pruner = pruner
        )

        logger.info(f"Initialized study: {self.study_name}")
        logger.info(f"Storage: {storage_url}")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation score to maximize
        
        """

        # Generate hyperparameters for this trial
        config = self._generate_trial_config(trial)

        # Create temporary directory for this trial
        with tempfile.TemporaryDirectory() as temp_dir:
            config['output_dir'] = temp_dir

            try:
                # Create data loaders
                train_loader, val_loader, _ = create_data_loaders(config)

                # Create trainer
                trainer = CompositionTrainer(config)

                # Train with early stopping for optimization
                best_score = self._train_with_pruning(trainer, train_loader,
                                                      val_loader, trial)
                
                return best_score
            
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned()
        
    def _generate_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Generate configuration for a single trial."""
        config = self.base_config.copy()

        # Optimize learning rates
        if 'learning_rate' in self.optimization_config:
            lr_config = self.optimization_config['learning_rate']
            backbone_lr = trial.suggest_float(
                'backbone_lr',
                lr_config.get('backbone_min', 1e-6),
                lr_config.get('backbone_max', 1e-3),
                log = True
            )

            head_lr = trial.suggest_float(
                'head_lr',
                lr_config.get('head_min', 1e-5),
                lr_config.get('head_max', 1e-2),
                log = True
            )

            config['optimizer']['backbone_lr'] = backbone_lr
            config['optimizer']['head_lr'] = head_lr

        # Optimize batch size
        if 'batch_size' in self.optimization_config:
            batch_config = self.optimization_config['batch_size']
            config['batch_size'] = trial.suggest_categorical(
                'batch_size',
                batch_config.get('choices', [8, 16, 32, 64])
            )

        # Optimize weight decay
        if 'weight_decay' in self.optimization_config:
            wd_config = self.optimization_config['weight_decay']
            config['optimizer']['weight_decay'] = trial.suggest_float(
                'weight_decay',
                wd_config.get('min', 1e-6),
                wd_config.get('max', 1e-2),
                log = True
            )
        
        # Optimize model architecture
        if 'model_architecture' in self.optimization_config:
            arch_config = self.optimization_config['model_architecture']
            
            if 'hidden_size' in arch_config:
                config['model']['hidden_size'] = trial.suggest_categorical(
                    'hidden_size',
                    arch_config['hidden_size'].get('choices', [384, 512, 768, 1024])
                )
            
            if 'num_attention_heads' in arch_config:
                config['model']['num_attention_heads'] = trial.suggest_categorical(
                    'num_attention_heads',
                    arch_config['num_attention_heads'].get('choices', [6, 8, 12, 16])
                )
            
            if 'num_hidden_layers' in arch_config:
                config['model']['num_hidden_layers'] = trial.suggest_int(
                    'num_hidden_layers',
                    arch_config['num_hidden_layers'].get('min', 6),
                    arch_config['num_hidden_layers'].get('max', 24)
                )
            
            if 'backbone' in arch_config:
                config['model']['backbone'] = trial.suggest_categorical(
                    'backbone',
                    arch_config['backbone'].get('choices', ['resnet50', 'resnet101', 'efficientnet_b0'])
                )

        # Optimize loss function weights
        if 'loss_weights' in self.optimization_config:
            weight_config = self.optimization_config['loss_weights']
            task_weights = {}
            
            for task in ['rule_of_thirds', 'leading_lines', 'symmetry', 'depth']:
                if task in weight_config:
                    task_weights[task] = trial.suggest_float(
                        f'{task}_weight',
                        weight_config[task].get('min', 0.1),
                        weight_config[task].get('max', 2.0)
                    )
            
            if task_weights:
                config['loss']['task_weights'] = task_weights
        
        # Optimzie data augmentation
        if 'augmentation' in self.optimization_config:
            aug_config = self.optimization_config['augmentation']

            # Color augmentation strength
            if 'color_jitter' in aug_config:
                config['data']['color_jitter_strength'] = trial.suggest_float(
                    'color_jitter_strength',
                    aug_config['color_jitter'].get('min', 0.0),
                    aug_config['color_jitter'].get('max', 0.5)
                )

            # Geometric augmentation
            if 'rotation' in aug_config:
                config['data']['rotation_degrees'] = trial.suggest_int(
                    'rotation_degrees',
                    aug_config['rotation'].get('min', 0),
                    aug_config['rotation'].get('max', 15)
                )
            
        # Optimize dropout rates
        if 'dropout' in self.optimization_config:
            dropout_config = self.optimization_config['dropout']
            config['model']['dropout'] = trial.suggest_float(
                'dropout',
                dropout_config.get('min', 0.0),
                dropout_config.get('max', 0.5)
            )
        
        # Optimize scheduler parameters
        if 'scheduler' in self.optimization_config:
            sched_config = self.optimization_config['scheduler']
            
            if config.get('scheduler', {}).get('type') == 'cosine':
                config['scheduler']['min_lr'] = trial.suggest_float(
                    'min_lr',
                    sched_config.get('min_lr_min', 1e-8),
                    sched_config.get('min_lr_max', 1e-5),
                    log=True
                )

        return config
    
    def _train_with_pruning(self, trainer: CompositionTrainer,
                            train_loader, val_loader, trial: optuna.Trial) -> float:
        """
        Train model with pruning for unpromising trials.
        
        Args:
            trainer: Composition trainer
            train_loader: Training data loader
            val_loader: Validation data loader
            trial: Optuna trial
            
        Returns:
            Best validation score achieved
        """

        best_score = 0.0
        patience_counter = 0
        patience = 5  # Early stopping patience for optimization

        for epoch in range(self.base_config['epochs']):
            trainer.current_epoch = epoch

            # Training step
            train_metrics = trainer.train_epoch(train_loader)

            # Validation step
            val_metrics = trainer.validate_epoch(val_loader)

            # Get current score (use overall_score or negative loss)
            current_score = val_metrics.get('overall_score', -val_metrics['total_loss'])

            # Update best score
            if current_score > best_score:
                best_score = current_score
                patience_counter = 0
            
            else:
                patience_counter += 1
            
            # Report intermediate value for pruning
            trial.report(current_score, epoch)

            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping for optimization
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return best_score
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Best hyperparameters found
        
        """

        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials = self.n_trials,
            show_progress_bar = True,
            catch = (Exception,)
        )

        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value

        logger.info(f"Optimization completed!")
        logger.info(f"Best score: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params:.4f}")

        # Save results
        self._save_optimization_results()

        # Generate best configuration
        best_config = self._generate_best_config(best_params)

        return best_config
    
    def _save_optimization_results(self):
        """Save optimization results and analysis."""

        # Save best parameters
        best_params_path = self.output_dir / 'best_parameters.json'
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'best_trial': self.study.best_trial.number 
            }, f, indent = 2)

        # Save all trials
        trials_path = self.output_dir / 'all_trials.json'
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })

        with open(trials_path, 'w'):
            json.dump(trials_data, f, ident = 2)
        
        # Generate optimization report
        self._generate_optimization_report()

    def _generate_optimization_report(self):
        """Generate detailed optimization report."""
        report_path = self.output_dir / 'optimization_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Hyperparameter Optimization Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Study Name: {self.study_name}\n")
            f.write(f"Total Trials: {len(self.study.trials)}\n")
            f.write(f"Completed Trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
            f.write(f"Pruned Trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
            f.write(f"Failed Trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}\n\n")
            
            f.write(f"Best Score: {self.study.best_value:.6f}\n")
            f.write(f"Best Trial: {self.study.best_trial.number}\n\n")
            
            f.write("Best Parameters:\n")
            for param, value in self.study.best_params.items():
                f.write(f"  {param}: {value}\n")
            
            # Parameter importance
            f.write("\n\nParameter Importance:\n")
            try:
                importance = optuna.importance.get_param_importances(self.study)
                for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {param}: {imp:.4f}\n")
            except Exception as e:
                f.write(f"  Could not compute importance: {e}\n")
    
    def _generate_best_config(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the best configuration from optimization results."""
        config = self.base_config.copy()
        
        # Apply best parameters to config
        for param, value in best_params.items():
            if param == 'backbone_lr':
                config['optimizer']['backbone_lr'] = value
            elif param == 'head_lr':
                config['optimizer']['head_lr'] = value
            elif param == 'batch_size':
                config['batch_size'] = value
            elif param == 'weight_decay':
                config['optimizer']['weight_decay'] = value
            elif param == 'hidden_size':
                config['model']['hidden_size'] = value
            elif param == 'num_attention_heads':
                config['model']['num_attention_heads'] = value
            elif param == 'num_hidden_layers':
                config['model']['num_hidden_layers'] = value
            elif param == 'backbone':
                config['model']['backbone'] = value
            elif param == 'dropout':
                config['model']['dropout'] = value
            elif param == 'min_lr':
                config['scheduler']['min_lr'] = value
            elif param.endswith('_weight'):
                task = param.replace('_weight', '')
                if 'task_weights' not in config['loss']:
                    config['loss']['task_weights'] = {}
                config['loss']['task_weights'][task] = value
            elif param == 'color_jitter_strength':
                config['data']['color_jitter_strength'] = value
            elif param == 'rotation_degrees':
                config['data']['rotation_degrees'] = value
        
        # Save best config
        best_config_path = self.output_dir / 'best_config.json'
        with open(best_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def continue_optimization(self, additional_trials: int):
        """
        Continue optimization with additional trials.
        
        Args:
            additional_trials: Number of additional trials to run
        """
        logger.info(f"Continuing optimization with {additional_trials} additional trials")
        
        self.study.optimize(
            self.objective,
            n_trials=additional_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        # Update results
        self._save_optimization_results()
        
        return self._generate_best_config(self.study.best_params)
    
def run_hyperparameter_optimization(base_config_path: str,
                                   optimization_config_path: str,
                                   n_trials: int = 100,
                                   study_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Run hyperparameter optimization from configuration files.
    
    Args:
        base_config_path: Path to base training configuration
        optimization_config_path: Path to optimization configuration
        n_trials: Number of optimization trials
        study_name: Name for the study
        
    Returns:
        Best configuration found
    """
    # Load configurations
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    with open(optimization_config_path, 'r') as f:
        optimization_config = json.load(f)
    
    # Create optimizer
    optimizer = CompositionHyperparameterOptimizer(
        base_config=base_config,
        optimization_config=optimization_config,
        n_trials=n_trials,
        study_name=study_name
    )
    
    # Run optimization
    best_config = optimizer.optimize()
    
    return best_config

class GridSearchOptimizer:
    """
    Grid search optimization for smaller parameter spaces.
    """
    
    def __init__(self, base_config: Dict[str, Any], grid_config: Dict[str, Any]):
        """
        Initialize grid search optimizer.
        
        Args:
            base_config: Base training configuration
            grid_config: Grid search parameter space
        """
        self.base_config = base_config
        self.grid_config = grid_config
        
        # Generate all parameter combinations
        self.parameter_combinations = self._generate_parameter_combinations()
        
        self.output_dir = Path(base_config['output_dir']) / 'grid_search'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        import itertools
        
        # Extract parameter names and values
        param_names = list(self.grid_config.keys())
        param_values = [self.grid_config[name] for name in param_names]
        
        # Generate all combinations
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def run_grid_search(self) -> Dict[str, Any]:
        """
        Run grid search optimization.
        
        Returns:
            Best configuration found
        """
        logger.info(f"Starting grid search with {len(self.parameter_combinations)} combinations")
        
        best_score = float('-inf')
        best_config = None
        results = []
        
        for i, params in enumerate(self.parameter_combinations):
            logger.info(f"Running combination {i+1}/{len(self.parameter_combinations)}: {params}")
            
            # Create config for this combination
            config = self._create_config_from_params(params)
            
            try:
                # Train and evaluate
                score = self._evaluate_config(config)
                
                results.append({
                    'params': params,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_config = config
                    
                logger.info(f"Score: {score:.4f} (Best: {best_score:.4f})")
                
            except Exception as e:
                logger.error(f"Failed combination {i+1}: {e}")
                results.append({
                    'params': params,
                    'score': None,
                    'error': str(e)
                })
        
        # Save results
        self._save_grid_search_results(results, best_config)
        
        return best_config
    
    def _create_config_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create full config from grid parameters."""
        config = self.base_config.copy()
        
        # Apply grid parameters (simple implementation)
        for param, value in params.items():
            keys = param.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        
        return config
    
    def _evaluate_config(self, config: Dict[str, Any]) -> float:
        """Evaluate a configuration and return validation score."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config['output_dir'] = temp_dir
            config['epochs'] = min(10, config.get('epochs', 50))  # Shorter training for grid search
            
            # Create data loaders
            train_loader, val_loader, _ = create_data_loaders(config)
            
            # Create and train model
            trainer = CompositionTrainer(config)
            trainer.train(train_loader, val_loader)
            
            # Return best validation score
            return trainer.best_val_score
    
    def _save_grid_search_results(self, results: List[Dict], best_config: Dict[str, Any]):
        """Save grid search results."""
        results_path = self.output_dir / 'grid_search_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'results': results,
                'best_config': best_config
            }, f, indent=2)
        
        best_config_path = self.output_dir / 'best_config.json'
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)