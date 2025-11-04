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




# optimize the model using jit and half precision
print("Loading VAE encoder...")
encoder = VAEImageEncoder(vae_config)

print("Testing inference time before optimization...")
# do multiple passes to check inference time
dummy_input = torch.randn(1, 1, vae_config.image_res[0], vae_config.image_res[1]).to("cuda")

with torch.no_grad():
    start_time = time.time()
    for _ in range(100):
        output = encoder(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 100.0
    # print(f"Average inference time before optimization: {avg_inference_time*1000.0:.2f} ms")

encoder.eval()
with torch.no_grad():
    start_time = time.time()
    for _ in range(100):
        output = encoder(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 100.0
    print(f"Average inference time before optimization: {avg_inference_time*1000.0:.2f} ms")

print("Optimizing VAE encoder...")
# freeze and optimize the model for very fast inference
encoder.eval()
encoder.half()
example_input = torch.randn(1, 1, vae_config.image_res[0], vae_config.image_res[1]).half().to("cuda")
optimized_encoder = torch.jit.trace(encoder, example_input)
optimized_encoder = torch.jit.optimize_for_inference(optimized_encoder)

print("Testing inference time after optimization...")
# do inference using optimized model
with torch.no_grad():
    start_time = time.time()
    for _ in range(100):
        output = optimized_encoder(dummy_input.half())
    torch.cuda.synchronize()
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 100.0
    print(f"Average inference time after optimization: {avg_inference_time*1000.0:.2f} ms")

