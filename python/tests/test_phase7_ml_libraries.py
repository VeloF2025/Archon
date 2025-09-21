"""
Test suite for Phase 7 ML Library Integrations
Validates TensorFlow, PyTorch, and Scikit-learn integrations
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import Dict, Any, List

# Import ML integration module
from src.agents.ml.production_ml_integration import (
    ProductionMLIntegration, ModelType, MLFramework,
    TrainingConfig, ModelMetrics
)


class TestProductionMLIntegration:
    """Test production ML library integrations"""
    
    @pytest.fixture
    def ml_integration(self):
        """Create ML integration instance"""
        return ProductionMLIntegration()
    
    def test_ml_integration_initialization(self, ml_integration):
        """Test ML integration initialization"""
        assert ml_integration is not None
        assert ml_integration.models == {}
        assert ml_integration.training_history == {}
    
    def test_tensorflow_availability_check(self, ml_integration):
        """Test TensorFlow availability detection"""
        tf_available = ml_integration._check_tensorflow()
        # Will be True if TensorFlow installed, False otherwise
        assert isinstance(tf_available, bool)
        
        if tf_available:
            print("TensorFlow is available for testing")
        else:
            print("TensorFlow not installed - using mock implementation")
    
    def test_pytorch_availability_check(self, ml_integration):
        """Test PyTorch availability detection"""
        pytorch_available = ml_integration._check_pytorch()
        assert isinstance(pytorch_available, bool)
        
        if pytorch_available:
            print("PyTorch is available for testing")
        else:
            print("PyTorch not installed - using mock implementation")
    
    def test_sklearn_availability_check(self, ml_integration):
        """Test Scikit-learn availability detection"""
        sklearn_available = ml_integration._check_sklearn()
        assert isinstance(sklearn_available, bool)
        
        if sklearn_available:
            print("Scikit-learn is available for testing")
        else:
            print("Scikit-learn not installed - using mock implementation")
    
    @pytest.mark.asyncio
    async def test_create_tensorflow_model(self, ml_integration):
        """Test TensorFlow model creation"""
        config = {
            "input_shape": (784,),
            "num_classes": 10,
            "hidden_layers": [128, 64],
            "activation": "relu",
            "optimizer": "adam"
        }
        
        model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.TENSORFLOW,
            config=config
        )
        
        assert model is not None
        assert "model_id" in model
        assert model["framework"] == MLFramework.TENSORFLOW
        assert model["type"] == ModelType.NEURAL_NETWORK
    
    @pytest.mark.asyncio
    async def test_create_pytorch_model(self, ml_integration):
        """Test PyTorch model creation"""
        config = {
            "input_size": 784,
            "hidden_size": 256,
            "output_size": 10,
            "num_layers": 2,
            "dropout": 0.2
        }
        
        model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.PYTORCH,
            config=config
        )
        
        assert model is not None
        assert "model_id" in model
        assert model["framework"] == MLFramework.PYTORCH
    
    @pytest.mark.asyncio
    async def test_create_sklearn_model(self, ml_integration):
        """Test Scikit-learn model creation"""
        config = {
            "algorithm": "random_forest",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        
        model = await ml_integration.create_model(
            model_type=ModelType.RANDOM_FOREST,
            framework=MLFramework.SKLEARN,
            config=config
        )
        
        assert model is not None
        assert "model_id" in model
        assert model["framework"] == MLFramework.SKLEARN
    
    @pytest.mark.asyncio
    async def test_train_model_with_mock_data(self, ml_integration):
        """Test model training with mock data"""
        # Create model
        model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.TENSORFLOW,
            config={"input_shape": (10,), "num_classes": 3}
        )
        
        # Create mock training data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 3, 20)
        
        # Training configuration
        train_config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.001,
            validation_split=0.2
        )
        
        # Train model
        history = await ml_integration.train_model(
            model_id=model["model_id"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=train_config
        )
        
        assert history is not None
        assert "loss" in history
        assert "epochs_completed" in history
        assert history["epochs_completed"] <= 5
    
    @pytest.mark.asyncio
    async def test_model_prediction(self, ml_integration):
        """Test model prediction"""
        # Create and train simple model
        model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.TENSORFLOW,
            config={"input_shape": (5,), "num_classes": 2}
        )
        
        # Mock training
        X_train = np.random.randn(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        await ml_integration.train_model(
            model_id=model["model_id"],
            X_train=X_train,
            y_train=y_train,
            config=TrainingConfig(epochs=2, batch_size=16)
        )
        
        # Test prediction
        X_test = np.random.randn(10, 5)
        predictions = await ml_integration.predict(
            model_id=model["model_id"],
            X=X_test
        )
        
        assert predictions is not None
        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions.flatten())
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, ml_integration):
        """Test model evaluation metrics"""
        # Create model
        model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.TENSORFLOW,
            config={"input_shape": (8,), "num_classes": 3}
        )
        
        # Mock data
        X_test = np.random.randn(30, 8)
        y_test = np.random.randint(0, 3, 30)
        
        # Evaluate
        metrics = await ml_integration.evaluate_model(
            model_id=model["model_id"],
            X_test=X_test,
            y_test=y_test
        )
        
        assert metrics is not None
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    @pytest.mark.asyncio
    async def test_model_save_and_load(self, ml_integration):
        """Test model saving and loading"""
        # Create model
        model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.PYTORCH,
            config={"input_size": 10, "hidden_size": 20, "output_size": 3}
        )
        
        # Save model
        save_path = f"/tmp/test_model_{model['model_id']}.pt"
        saved = await ml_integration.save_model(
            model_id=model["model_id"],
            path=save_path
        )
        assert saved["status"] == "saved"
        
        # Load model
        loaded = await ml_integration.load_model(
            path=save_path,
            framework=MLFramework.PYTORCH
        )
        assert loaded is not None
        assert "model_id" in loaded
    
    @pytest.mark.asyncio
    async def test_ensemble_model_creation(self, ml_integration):
        """Test ensemble model creation with multiple frameworks"""
        # Create multiple models
        models = []
        
        # TensorFlow model
        tf_model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.TENSORFLOW,
            config={"input_shape": (10,), "num_classes": 3}
        )
        models.append(tf_model["model_id"])
        
        # Scikit-learn model
        sklearn_model = await ml_integration.create_model(
            model_type=ModelType.RANDOM_FOREST,
            framework=MLFramework.SKLEARN,
            config={"n_estimators": 50}
        )
        models.append(sklearn_model["model_id"])
        
        # Create ensemble
        ensemble = await ml_integration.create_ensemble(
            model_ids=models,
            ensemble_type="voting",
            weights=[0.6, 0.4]
        )
        
        assert ensemble is not None
        assert ensemble["type"] == "ensemble"
        assert len(ensemble["models"]) == 2
    
    @pytest.mark.asyncio
    async def test_hyperparameter_optimization(self, ml_integration):
        """Test hyperparameter optimization"""
        # Define search space
        param_space = {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64],
            "hidden_layers": [[64], [128, 64], [256, 128, 64]]
        }
        
        # Run optimization
        best_params = await ml_integration.optimize_hyperparameters(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.TENSORFLOW,
            param_space=param_space,
            X_train=np.random.randn(100, 10),
            y_train=np.random.randint(0, 3, 100),
            n_trials=5
        )
        
        assert best_params is not None
        assert "learning_rate" in best_params
        assert "batch_size" in best_params
        assert best_params["score"] is not None
    
    @pytest.mark.asyncio
    async def test_transfer_learning(self, ml_integration):
        """Test transfer learning capabilities"""
        # Create base model
        base_model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.TENSORFLOW,
            config={"input_shape": (224, 224, 3), "num_classes": 1000}
        )
        
        # Apply transfer learning
        transfer_model = await ml_integration.apply_transfer_learning(
            base_model_id=base_model["model_id"],
            new_num_classes=10,
            freeze_base_layers=True,
            fine_tune_layers=["fc", "classifier"]
        )
        
        assert transfer_model is not None
        assert transfer_model["base_model"] == base_model["model_id"]
        assert transfer_model["num_classes"] == 10
    
    @pytest.mark.asyncio
    async def test_distributed_training_setup(self, ml_integration):
        """Test distributed training configuration"""
        # Configure distributed training
        dist_config = await ml_integration.setup_distributed_training(
            framework=MLFramework.PYTORCH,
            num_gpus=2,
            num_nodes=1,
            strategy="data_parallel"
        )
        
        assert dist_config is not None
        assert dist_config["strategy"] == "data_parallel"
        assert "device_map" in dist_config
    
    def test_gpu_acceleration_detection(self, ml_integration):
        """Test GPU availability detection"""
        gpu_info = ml_integration.get_gpu_info()
        
        assert gpu_info is not None
        assert "cuda_available" in gpu_info
        assert "device_count" in gpu_info
        
        if gpu_info["cuda_available"]:
            assert gpu_info["device_count"] > 0
            print(f"Found {gpu_info['device_count']} GPU(s)")
        else:
            print("No GPU available - using CPU")
    
    @pytest.mark.asyncio
    async def test_model_export_formats(self, ml_integration):
        """Test model export to different formats"""
        # Create model
        model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.TENSORFLOW,
            config={"input_shape": (10,), "num_classes": 3}
        )
        
        # Export to ONNX
        onnx_export = await ml_integration.export_model(
            model_id=model["model_id"],
            format="onnx",
            path="/tmp/test_model.onnx"
        )
        assert onnx_export["format"] == "onnx"
        
        # Export to TorchScript
        if ml_integration._check_pytorch():
            torch_export = await ml_integration.export_model(
                model_id=model["model_id"],
                format="torchscript",
                path="/tmp/test_model.pt"
            )
            assert torch_export["format"] == "torchscript"
    
    @pytest.mark.asyncio
    async def test_model_quantization(self, ml_integration):
        """Test model quantization for deployment"""
        # Create model
        model = await ml_integration.create_model(
            model_type=ModelType.NEURAL_NETWORK,
            framework=MLFramework.TENSORFLOW,
            config={"input_shape": (10,), "num_classes": 3}
        )
        
        # Quantize model
        quantized = await ml_integration.quantize_model(
            model_id=model["model_id"],
            quantization_type="int8",
            calibration_data=np.random.randn(100, 10)
        )
        
        assert quantized is not None
        assert quantized["quantization"] == "int8"
        assert quantized["size_reduction"] > 0
    
    @pytest.mark.asyncio
    async def test_automl_pipeline(self, ml_integration):
        """Test AutoML pipeline"""
        # Mock dataset
        X = np.random.randn(200, 15)
        y = np.random.randint(0, 3, 200)
        
        # Run AutoML
        automl_result = await ml_integration.run_automl(
            X=X,
            y=y,
            task_type="classification",
            time_budget=60,  # 1 minute
            metric="accuracy"
        )
        
        assert automl_result is not None
        assert "best_model" in automl_result
        assert "best_score" in automl_result
        assert automl_result["best_score"] > 0


@pytest.mark.integration
class TestMLFrameworkIntegration:
    """Integration tests for ML frameworks working together"""
    
    @pytest.mark.asyncio
    async def test_multi_framework_ensemble(self):
        """Test ensemble with models from different frameworks"""
        ml_integration = ProductionMLIntegration()
        
        # Create models from different frameworks
        models = []
        
        # TensorFlow model
        if ml_integration._check_tensorflow():
            tf_model = await ml_integration.create_model(
                model_type=ModelType.NEURAL_NETWORK,
                framework=MLFramework.TENSORFLOW,
                config={"input_shape": (20,), "num_classes": 3}
            )
            models.append(tf_model)
        
        # PyTorch model
        if ml_integration._check_pytorch():
            pt_model = await ml_integration.create_model(
                model_type=ModelType.NEURAL_NETWORK,
                framework=MLFramework.PYTORCH,
                config={"input_size": 20, "hidden_size": 50, "output_size": 3}
            )
            models.append(pt_model)
        
        # Scikit-learn model
        if ml_integration._check_sklearn():
            sk_model = await ml_integration.create_model(
                model_type=ModelType.RANDOM_FOREST,
                framework=MLFramework.SKLEARN,
                config={"n_estimators": 100}
            )
            models.append(sk_model)
        
        if len(models) > 1:
            # Create ensemble
            ensemble = await ml_integration.create_ensemble(
                model_ids=[m["model_id"] for m in models],
                ensemble_type="stacking"
            )
            
            assert ensemble is not None
            assert len(ensemble["models"]) == len(models)
            
            # Test ensemble prediction
            X_test = np.random.randn(10, 20)
            predictions = await ml_integration.predict_ensemble(
                ensemble_id=ensemble["ensemble_id"],
                X=X_test
            )
            
            assert predictions is not None
            assert len(predictions) == 10
    
    @pytest.mark.asyncio
    async def test_model_conversion_between_frameworks(self):
        """Test model conversion between frameworks"""
        ml_integration = ProductionMLIntegration()
        
        # Create TensorFlow model
        if ml_integration._check_tensorflow():
            tf_model = await ml_integration.create_model(
                model_type=ModelType.NEURAL_NETWORK,
                framework=MLFramework.TENSORFLOW,
                config={"input_shape": (10,), "num_classes": 2}
            )
            
            # Convert to ONNX
            onnx_path = "/tmp/tf_model.onnx"
            await ml_integration.export_model(
                model_id=tf_model["model_id"],
                format="onnx",
                path=onnx_path
            )
            
            # Load in PyTorch if available
            if ml_integration._check_pytorch():
                pt_model = await ml_integration.import_onnx_model(
                    path=onnx_path,
                    framework=MLFramework.PYTORCH
                )
                assert pt_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])