import torch
from model import NeRFNetwork

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = NeRFNetwork(bound=2, cuda_ray=False).to(device)
    print("Model initialized successfully!")

    # Dummy inputs
    N = 1024
    x = torch.rand(N, 3, device=device) * 4 - 2 # in [-2, 2]
    d = torch.randn(N, 3, device=device)
    d = d / torch.norm(d, dim=-1, keepdim=True)

    # Forward pass
    with torch.no_grad():
        sigma, color = model(x, d)
    
    print(f"Forward pass completed!")
    print(f"Sigma shape: {sigma.shape}, Color shape: {color.shape}")
    print(f"Sigma min/max: {sigma.min().item():.4f}, {sigma.max().item():.4f}")
    print(f"Color min/max: {color.min().item():.4f}, {color.max().item():.4f}")

if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
