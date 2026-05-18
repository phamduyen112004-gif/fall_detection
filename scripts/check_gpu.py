"""
GPU Check Script
Run: python scripts/check_gpu.py
"""

from src.utils import check_gpu


def main():
    print("=" * 50)
    print("GPU CHECK")
    print("=" * 50)
    
    info = check_gpu()
    
    print(f"\nCUDA Available: {info['cuda_available']}")
    print(f"Device:        {info['device']}")
    
    if info['cuda_available']:
        print(f"GPU Name:      {info['gpu_name']}")
        print(f"GPU Memory:    {info['gpu_memory']:.2f} GB")
    else:
        print("\nRunning on CPU - install CUDA for GPU acceleration")
    
    print("=" * 50)


if __name__ == "__main__":
    main()
