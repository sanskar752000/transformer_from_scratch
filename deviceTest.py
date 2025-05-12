import torch

# Check if CUDA (GPU) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using mps: {device.type}")
    print(torch.__version__)
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print (x)
# else:
#     print ("MPS device not found.")