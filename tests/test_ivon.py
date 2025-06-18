import torch
import pytest
from ivon import IVON
from accelerate import Accelerator


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model_and_data():
    torch.manual_seed(42)
    model = SimpleNet()
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    return model, X, y


def test_ivon_initialization():
    """Test basic IVON optimizer initialization"""
    model = SimpleNet()
    optimizer = IVON(
        model.parameters(),
        lr=0.1,
        ess=20,  # effective sample size
        mc_samples=1,
    )

    # Check key attributes are set correctly
    assert optimizer.mc_samples == 1
    assert optimizer.current_step == 0
    assert optimizer.hess_approx == "price"


def test_ivon_with_accelerate_gradient_accumulation(model_and_data):
    """Test IVON optimizer with Accelerate gradient accumulation"""
    model, X, y = model_and_data

    # Move input data to a consistent device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    model = model.to(device)

    # Accelerate setup
    accelerator = Accelerator(gradient_accumulation_steps=2)

    criterion = torch.nn.CrossEntropyLoss()
    base_optimizer = IVON(model.parameters(), lr=0.1, ess=len(X), mc_samples=1)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=2)

    # Prepare with Accelerate
    model, optimizer, dataloader = accelerator.prepare(
        model, base_optimizer, dataloader
    )

    model.train()

    # Store initial parameters and state
    initial_params = [p.clone() for p in model.parameters()]

    base_opt = optimizer.optimizer

    initial_state = {
        "count": base_opt.current_step,
        "avg_grad": base_opt.state.get("avg_grad", None).clone()
        if base_opt.state.get("avg_grad") is not None
        else None,
        "avg_nxg": base_opt.state.get("avg_nxg", None).clone()
        if base_opt.state.get("avg_nxg") is not None
        else None,
        "avg_gsq": base_opt.state.get("avg_gsq", None).clone()
        if base_opt.state.get("avg_gsq") is not None
        else None,
    }

    # Keep track of parameters
    batch_step_count = 0

    for _ in range(2):
        for _, (batch_x, batch_y) in enumerate(dataloader):
            with accelerator.accumulate(model):
                with optimizer.optimizer.sampled_params(train=True):
                    # Forward pass
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    # Backward pass
                    accelerator.backward(loss)

                # Call step when sync_gradients is True
                if accelerator.sync_gradients:
                    optimizer.step()
                    batch_step_count += 1

    # Check that parameters have been updated
    updated_params = list(model.parameters())
    any_param_updated = any(
        not torch.equal(init_p, updated_p)
        for init_p, updated_p in zip(initial_params, updated_params)
    )
    assert any_param_updated, "Parameters were not updated during training"

    # Check state has been updated
    current_count = base_opt.current_step
    initial_count = initial_state["count"]

    # Ensure step has been called
    assert current_count > initial_count, "Optimizer step count not incremented"

    # Check grad-related states have been updated
    final_avg_grad = base_opt.state.get("avg_grad")
    final_avg_nxg = base_opt.state.get("avg_nxg")
    final_avg_gsq = base_opt.state.get("avg_gsq")

    assert final_avg_grad is not None, "avg_grad not initialized"
    assert final_avg_nxg is not None, "avg_nxg not initialized"
    if base_opt.hess_approx == "gradsq":
        assert final_avg_gsq is not None, "avg_gsq not initialized"


def test_pytorch_native_gradient_accumulation(model_and_data):
    """Test IVON optimizer with PyTorch native gradient accumulation"""
    model, X, y = model_and_data

    # Move input data to a consistent device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = IVON(model.parameters(), lr=0.1, ess=len(X), mc_samples=1)

    # Gradient accumulation parameters
    accumulation_steps = 2

    # Store initial parameters and state
    initial_params = [p.clone() for p in model.parameters()]
    initial_state = {
        "count": optimizer.state.get("count", None),
        "avg_grad": optimizer.state.get("avg_grad", None),
        "avg_nxg": optimizer.state.get("avg_nxg", None),
        "avg_gsq": optimizer.state.get("avg_gsq", None),
    }

    model.train()
    running_loss = 0.0

    for _ in range(2):
        for i in range(len(X) // 2):
            # Select a subset of data
            batch_x = X[i * 2 : (i + 1) * 2]
            batch_y = y[i * 2 : (i + 1) * 2]

            with optimizer.sampled_params(train=True):
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y) / accumulation_steps

                # Backward pass
                loss.backward()
                running_loss += loss.item()

            # Update weights only after accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                running_loss = 0.0

    # Check that parameters have been updated
    updated_params = list(model.parameters())
    any_param_updated = any(
        not torch.equal(init_p, updated_p)
        for init_p, updated_p in zip(initial_params, updated_params)
    )
    assert any_param_updated, "Parameters were not updated during training"

    # Check state has been updated

    # Instead of checking state['count'], check current_step
    assert optimizer.current_step > initial_state["count"], (
        "Optimizer step count not incremented"
    )

    # Check grad-related states have been updated (they start as None, so should not be None now)
    base_opt = optimizer

    def validate_state(state_name):
        # Get the state value
        state_value = base_opt.state.get(state_name)

        # Validate the state value
        assert state_value is not None, f"{state_name} not initialized"
        assert not torch.all(state_value == 0), f"{state_name} is a zero tensor"
        assert state_value.numel() > 0, f"{state_name} is empty"

    validate_state("avg_grad")
    validate_state("avg_nxg")

    if base_opt.hess_approx == "gradsq":
        validate_state("avg_gsq")


def test_ivon_sampling(model_and_data):
    """Test parameter sampling functionality"""
    model, X, y = model_and_data

    optimizer = IVON(model.parameters(), lr=0.1, ess=len(X), mc_samples=3)

    # Store original parameters
    original_params = [p.clone() for p in model.parameters()]

    # Collect sampled parameters
    sampled_params_collection = []

    for _ in range(5):
        with optimizer.sampled_params():
            # Collect current parameter values
            current_params = [p.clone() for p in model.parameters()]
            sampled_params_collection.append(current_params)

    # Verify that sampling introduces variation
    variations = [
        not torch.equal(original_params[i], sampled_params_collection[0][i])
        for i in range(len(original_params))
    ]
    assert any(variations), "Parameter sampling did not introduce variations"


def test_ivon_hess_approx_methods():
    """Test different Hessian approximation methods"""
    model = SimpleNet()

    # Test 'price' method (default)
    optimizer_price = IVON(model.parameters(), lr=0.1, ess=20, hess_approx="price")
    assert optimizer_price.hess_approx == "price"

    # Test 'gradsq' method
    optimizer_gradsq = IVON(model.parameters(), lr=0.1, ess=20, hess_approx="gradsq")
    assert optimizer_gradsq.hess_approx == "gradsq"

    # Ensure invalid method raises ValueError
    with pytest.raises(ValueError):
        IVON(model.parameters(), lr=0.1, ess=20, hess_approx="invalid_method")


def test_ivon_device_move():
    """Test IVON optimizer when model is moved to different device after initialization"""
    model = SimpleNet()

    # Initialize optimizer on CPU
    optimizer = IVON(model.parameters(), lr=0.1, ess=20, mc_samples=1)

    # Move model to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model = model.to(device)

        # This should work without device mismatch errors
        with optimizer.sampled_params():
            # Collect current parameter values
            current_params = [p.clone() for p in model.parameters()]

        # Test training step as well
        X = torch.randn(10, 10).to(device)
        y = torch.randint(0, 2, (10,)).to(device)
        criterion = torch.nn.CrossEntropyLoss()

        with optimizer.sampled_params(train=True):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

        # This should also work
        optimizer.step()
    else:
        pytest.skip("CUDA not available")


def test_ivon_mixed_precision():
    """Test IVON optimizer with mixed precision (bfloat16)"""
    model = SimpleNet()

    # Convert model to bfloat16
    model = model.to(dtype=torch.bfloat16)

    optimizer = IVON(model.parameters(), lr=0.1, ess=20, mc_samples=1)

    X = torch.randn(10, 10, dtype=torch.bfloat16)
    y = torch.randint(0, 2, (10,))
    criterion = torch.nn.CrossEntropyLoss()

    # Store original parameter dtypes
    original_dtypes = {id(p): p.dtype for p in model.parameters()}

    # Test multiple training steps to ensure dtype consistency
    for step in range(3):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

        # Check gradients after sampled_params processing
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.dtype == param.dtype, (
                    f"Step {step}: Gradient dtype {param.grad.dtype} doesn't match parameter dtype {param.dtype}"
                )

        # Verify parameters still have correct dtype
        for param in model.parameters():
            assert param.dtype == original_dtypes[id(param)], (
                f"Step {step}: Parameter dtype changed from {original_dtypes[id(param)]} to {param.dtype}"
            )

        # Step the optimizer
        optimizer.step()

        # Final check after step
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.dtype == param.dtype, (
                    f"Step {step} after step(): Gradient dtype {param.grad.dtype} doesn't match parameter dtype {param.dtype}"
                )


def test_ivon_autocast_mixed_precision():
    """Test IVON optimizer with autocast mixed precision"""
    model = SimpleNet()
    optimizer = IVON(model.parameters(), lr=0.1, ess=20, mc_samples=1)

    X = torch.randn(10, 10)
    y = torch.randint(0, 2, (10,))
    criterion = torch.nn.CrossEntropyLoss()

    # Store original parameter dtypes
    original_dtypes = {id(p): p.dtype for p in model.parameters()}

    # Test training with autocast (simulates mixed precision training)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

    # Check that gradients maintain consistency
    for param in model.parameters():
        if param.grad is not None:
            # Gradients might be in different precision due to autocast
            assert param.grad.device == param.device, (
                f"Gradient device {param.grad.device} doesn't match parameter device {param.device}"
            )

    # Verify parameters still have correct dtype
    for param in model.parameters():
        assert param.dtype == original_dtypes[id(param)], (
            f"Parameter dtype changed from {original_dtypes[id(param)]} to {param.dtype}"
        )

    # Step the optimizer
    optimizer.step()

    # Final check - parameters should maintain their original dtype
    for param in model.parameters():
        assert param.dtype == original_dtypes[id(param)], (
            f"After step: Parameter dtype changed from {original_dtypes[id(param)]} to {param.dtype}"
        )


def test_ivon_mixed_precision_consistency():
    """Test IVON optimizer maintains dtype consistency in mixed precision scenarios"""
    model = SimpleNet()

    # Start with bfloat16 model
    model = model.to(dtype=torch.bfloat16)

    optimizer = IVON(model.parameters(), lr=0.1, ess=20, mc_samples=1)

    X = torch.randn(10, 10, dtype=torch.bfloat16)
    y = torch.randint(0, 2, (10,))
    criterion = torch.nn.CrossEntropyLoss()

    # Multiple training steps to test consistency
    for step in range(5):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

        # Check gradients and parameters maintain consistency
        for param in model.parameters():
            assert param.dtype == torch.bfloat16, (
                f"Step {step}: Parameter dtype changed to {param.dtype}"
            )
            if param.grad is not None:
                # Note: gradients might be promoted for stability, but parameters should maintain their dtype
                assert param.device == param.grad.device, (
                    f"Step {step}: Device mismatch"
                )

        optimizer.step()

        # After step, parameters should still be bfloat16
        for param in model.parameters():
            assert param.dtype == torch.bfloat16, (
                f"Step {step} after step(): Parameter dtype changed to {param.dtype}"
            )
