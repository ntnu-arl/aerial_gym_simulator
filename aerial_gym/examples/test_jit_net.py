# import the jit model and run inference
import torch, torch.jit
traced_model = torch.jit.load("traced_vae_encoder.pt").to("cuda:0").eval()
# create a random input tensor with batch size 4 and image size 270x480
input_tensor = torch.rand(32, 270, 480).to("cuda:0").half()
latent_means = traced_model(input_tensor)
print(latent_means[0])