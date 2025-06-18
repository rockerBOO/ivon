import torch
import numpy as np
import pytest
from ivon import IVON

class RegressionNet(torch.nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 50)
        self.fc2 = torch.nn.Linear(50, output_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def generate_regression_data(n_samples=200, input_dim=10, noise_std=0.1):
    """Generate synthetic regression data"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # True underlying function (non-linear)
    true_weights1 = np.random.randn(input_dim, 50)
    true_weights2 = np.random.randn(50, 1)
    true_bias1 = np.random.randn(50)
    true_bias2 = np.random.randn(1)
    
    X = torch.randn(n_samples, input_dim)
    
    # Non-linear transformation
    hidden = torch.relu(torch.matmul(X, torch.tensor(true_weights1, dtype=torch.float32)) + torch.tensor(true_bias1, dtype=torch.float32))
    y = torch.matmul(hidden, torch.tensor(true_weights2, dtype=torch.float32)) + torch.tensor(true_bias2, dtype=torch.float32)
    
    # Add some noise
    y += torch.randn_like(y) * noise_std
    
    return X, y

def test_ivon_sampling_variance():
    """Test that IVON introduces parameter sampling variation"""
    model = RegressionNet()
    optimizer = IVON(
        model.parameters(), 
        lr=0.01, 
        ess=100,
        mc_samples=5
    )
    
    # Store initial parameters
    initial_params = [p.clone() for p in model.parameters()]
    
    # Collect sampled parameters
    sampled_params_collection = []
    
    for _ in range(10):
        with optimizer.sampled_params():
            # Collect current parameter values
            current_params = [p.clone() for p in model.parameters()]
            sampled_params_collection.append(current_params)
    
    # Verify sampling introduces variation
    def param_variation(orig_params, sampled_params):
        return any(
            not torch.equal(orig_p, sampled_p) 
            for orig_p, sampled_p in zip(orig_params, sampled_params)
        )
    
    variations = [
        param_variation(initial_params, sampled_params) 
        for sampled_params in sampled_params_collection
    ]
    
    assert any(variations), "IVON sampling did not introduce parameter variations"

def test_ivon_convergence_comparison():
    """Compare IVON optimizer convergence with Adam optimizer"""
    # Prepare data
    X, y = generate_regression_data(noise_std=0.5)
    
    # Setup models
    def train_model(optimizer_class, **optimizer_kwargs):
        torch.manual_seed(42)  # Ensure reproducibility
        model = RegressionNet()
        criterion = torch.nn.MSELoss()
        
        # Choose optimizer with adaptive hyperparameters
        if optimizer_class == IVON:
            optimizer = optimizer_class(
                model.parameters(), 
                lr=0.001,
                ess=len(X),
                mc_samples=5,
                weight_decay=1e-4,
                hess_init=10.0,
                **optimizer_kwargs
            )
        else:
            optimizer = optimizer_class(
                model.parameters(), 
                lr=0.001,
                weight_decay=1e-4,
                **optimizer_kwargs
            )
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        losses = []
        
        for epoch in range(300):
            # Adaptive learning rate decay
            if epoch > 0 and epoch % 50 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            
            def closure():
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                return loss
            
            # For IVON, use sampled params
            if optimizer_class == IVON:
                with optimizer.sampled_params(train=True):
                    loss = closure()
                    optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
            
            current_loss = loss.item()
            losses.append(current_loss)
            
            # Early stopping
            if current_loss < best_loss * 0.99:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        return model, losses
    
    # Train with different optimizers
    ivon_model, ivon_losses = train_model(IVON)
    adam_model, adam_losses = train_model(torch.optim.Adam)
    
    # Evaluate models
    def evaluate_model(model, X, y):
        with torch.no_grad():
            outputs = model(X)
            mse = torch.nn.functional.mse_loss(outputs, y)
            return mse.item()
    
    ivon_mse = evaluate_model(ivon_model, X, y)
    adam_mse = evaluate_model(adam_model, X, y)
    
    # Validate learning progression
    def validate_learning_progression(losses):
        loss_array = np.array(losses)
        loss_reduction_rates = np.polyfit(np.arange(len(loss_array)), loss_array, 1)
        return loss_reduction_rates[0]  # Slope of the linear fit
    
    # Compute learning progression
    ivon_learning_slope = validate_learning_progression(ivon_losses)
    adam_learning_slope = validate_learning_progression(adam_losses)
    
    # Ensure meaningful learning is happening
    assert ivon_learning_slope < 0, "IVON failed to show loss reduction"
    assert adam_learning_slope < 0, "Adam failed to show loss reduction"
    
    # Performance validation - IVON should perform reasonably compared to Adam
    max_acceptable_mse = adam_mse * 10  # Allow IVON to be up to 10x worse than Adam
    assert ivon_mse <= max_acceptable_mse, \
        f"IVON optimizer failed to converge within acceptable range (MSE: {ivon_mse}, Max Acceptable: {max_acceptable_mse})"
    
    # Deviation check
    performance_deviation = abs(ivon_mse - adam_mse) / (adam_mse + 1e-7)
    assert performance_deviation < 5.0, \
        f"IVON performance deviates too much from Adam (Deviation: {performance_deviation:.4f})"

def test_ivon_uncertainty_estimation():
    """Test IVON's uncertainty estimation capabilities"""
    X, y = generate_regression_data()
    
    model = RegressionNet()
    optimizer = IVON(
        model.parameters(), 
        lr=0.01, 
        ess=len(X),
        mc_samples=10
    )
    
    # Train the model
    criterion = torch.nn.MSELoss()
    for _ in range(100):
        def closure():
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            return loss
        
        with optimizer.sampled_params(train=True):
            loss = closure()
            optimizer.step(closure)
    
    # Collect multiple predictions
    sample_predictions = []
    for _ in range(50):
        with optimizer.sampled_params():
            with torch.no_grad():
                pred = model(X)
                sample_predictions.append(pred)
    
    # Convert to numpy for analysis
    sample_predictions = torch.stack(sample_predictions).numpy()
    
    # Compute prediction variance
    pred_variance = np.var(sample_predictions, axis=0)
    
    # Check variance characteristics
    assert pred_variance.mean() > 0, "IVON failed to capture prediction uncertainty"
    assert np.all(pred_variance < 1.0), "Prediction variance too high"

def test_ivon_gradient_accumulation_robustness():
    """Test IVON optimizer's robustness with gradient accumulation"""
    X, y = generate_regression_data()
    
    model = RegressionNet()
    optimizer = IVON(
        model.parameters(), 
        lr=0.01, 
        ess=len(X),
        mc_samples=1
    )
    
    # Simulate gradient accumulation manually
    criterion = torch.nn.MSELoss()
    running_loss = 0.0
    accumulation_steps = 4
    
    initial_params = [p.clone() for p in model.parameters()]
    
    for epoch in range(2):
        for i in range(0, len(X), accumulation_steps):
            batch_x = X[i:i+accumulation_steps]
            batch_y = y[i:i+accumulation_steps]
            
            # Simulate gradient accumulation
            batch_loss = criterion(model(batch_x), batch_y) / accumulation_steps
            batch_loss.backward()
            running_loss += batch_loss.item()
            
            # Step only after accumulation
            if (i + accumulation_steps) % len(X) == 0:
                def closure():
                    return running_loss
                
                optimizer.step(closure)
                optimizer.zero_grad()
                running_loss = 0.0
    
    # Check parameters were updated
    updated_params = list(model.parameters())
    any_updated = any(
        not torch.equal(init_p, updated_p) 
        for init_p, updated_p in zip(initial_params, updated_params)
    )
    assert any_updated, "Parameters not updated during gradient accumulation"

def test_ivon_multi_device_support():
    """Test IVON optimizer's multi-device support"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create models on different devices
    cpu_model = RegressionNet()
    cuda_model = RegressionNet().cuda()
    
    X_cpu = torch.randn(100, 10)
    y_cpu = torch.randn(100, 1)
    
    X_cuda = X_cpu.cuda()
    y_cuda = y_cpu.cuda()
    
    # Test CPU training
    cpu_optimizer = IVON(
        cpu_model.parameters(), 
        lr=0.01, 
        ess=len(X_cpu),
        mc_samples=1
    )
    
    # Test CUDA training
    cuda_optimizer = IVON(
        cuda_model.parameters(), 
        lr=0.01, 
        ess=len(X_cuda),
        mc_samples=1
    )
    
    # Train both
    criterion = torch.nn.MSELoss()
    
    def train_model(model, optimizer, X, y):
        for _ in range(50):
            def closure():
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                return loss
            
            with optimizer.sampled_params(train=True):
                loss = closure()
                optimizer.step(closure)
        return model
    
    # Ensure no exceptions are raised
    try:
        train_model(cpu_model, cpu_optimizer, X_cpu, y_cpu)
        train_model(cuda_model, cuda_optimizer, X_cuda, y_cuda)
    except Exception as e:
        pytest.fail(f"Multi-device training failed: {e}")
