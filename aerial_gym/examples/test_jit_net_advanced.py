from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder
import torch
import time
import torch.nn as nn
from aerial_gym import AERIAL_GYM_DIRECTORY

class vae_config:
    latent_dims = 64
    model_file = (
        AERIAL_GYM_DIRECTORY
        + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
    )
    model_folder = AERIAL_GYM_DIRECTORY
    image_res = (270, 480)
    interpolation_mode = "nearest"
    return_sampled_latent = False

def create_tensorrt_model(model, input_tensor):
    """Create TensorRT optimized model if available"""
    try:
        import torch_tensorrt
        
        # Compile with TensorRT
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[input_tensor],
            enabled_precisions={torch.float, torch.half},
            workspace_size=1 << 30,  # 1GB
        )
        print("TensorRT compilation successful!")
        return trt_model
    except ImportError:
        print("TensorRT not available, skipping TensorRT optimization")
        return None
    except Exception as e:
        print(f"TensorRT compilation failed: {e}")
        return None

def optimize_model_with_fusion(model):
    """Apply graph fusion optimizations"""
    # Freeze the model to enable more optimizations
    model = torch.jit.freeze(model)
    
    # Apply optimization passes
    torch.jit.optimize_for_inference(model)
    
    return model

def benchmark_with_profiling(model, input_tensor, num_iterations=100):
    """Benchmark with CUDA profiling"""
    torch.cuda.synchronize()
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()
    
    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return prof

def main():
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load model
    encoder = VAEImageEncoder(config=vae_config).to("cuda:0")
    encoder.eval()
    
    # Check if we can use mixed precision
    use_mixed_precision = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    
    # Create input tensor
    input_tensor = torch.randn(1, 1, 270, 480).to("cuda:0")
    
    if use_mixed_precision:
        print("Using mixed precision (FP16)")
        encoder = encoder.half()
        input_tensor = input_tensor.half()
    
    # Method 1: Standard JIT tracing
    print("=== Creating JIT traced model ===")
    traced_model = torch.jit.trace(encoder, input_tensor)
    traced_model = optimize_model_with_fusion(traced_model)
    
    # Method 2: TensorRT optimization (if available)
    print("\n=== Attempting TensorRT optimization ===")
    trt_model = create_tensorrt_model(traced_model, input_tensor)
    
    # Method 3: Torch-TensorRT (alternative)
    print("\n=== Testing Torch Compile (PyTorch 2.0+) ===")
    try:
        compiled_model = torch.compile(encoder, mode="max-autotune")
        print("Torch compile successful!")
    except Exception as e:
        print(f"Torch compile failed: {e}")
        compiled_model = None
    
    # Benchmark all models
    models_to_test = [
        ("Original", encoder),
        ("JIT Traced + Optimized", traced_model),
    ]
    
    if trt_model is not None:
        models_to_test.append(("TensorRT", trt_model))
    
    if compiled_model is not None:
        models_to_test.append(("Torch Compiled", compiled_model))
    
    print("\n=== Benchmarking Results ===")
    results = {}
    
    for name, model in models_to_test:
        print(f"\nTesting {name}...")
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = model(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        fps = 100 / elapsed_time
        
        results[name] = fps
        print(f"{name}: {fps:.2f} FPS ({elapsed_time/100*1000:.2f} ms per inference)")
    
    # Print speedup comparison
    baseline_fps = results["Original"]
    print("\n=== Speedup Comparison ===")
    for name, fps in results.items():
        speedup = fps / baseline_fps
        print(f"{name}: {speedup:.2f}x speedup")
    
    # Test with different batch sizes
    print("\n=== Batch Size Analysis ===")
    best_model = traced_model if trt_model is None else trt_model
    
    for batch_size in [1, 2, 4, 8, 16]:
        try:
            batch_input = torch.randn(batch_size, 1, 270, 480).to("cuda:0")
            if use_mixed_precision:
                batch_input = batch_input.half()
            
            # Warm up
            for _ in range(5):
                with torch.no_grad():
                    _ = best_model(batch_input)
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(50):
                    _ = best_model(batch_input)
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_samples = 50 * batch_size
            fps_per_sample = total_samples / (end_time - start_time)
            
            print(f"Batch size {batch_size}: {fps_per_sample:.2f} FPS per sample")
            
        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {batch_size}: Out of memory")
            break
    
    # Profile the best model
    print("\n=== Profiling Best Model ===")
    benchmark_with_profiling(best_model, input_tensor, 50)

if __name__ == "__main__":
    main()
