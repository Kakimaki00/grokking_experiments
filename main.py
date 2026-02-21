import argparse
import torch
import gc
from loaders import get_dataloaders
from training import Trainer
from models import WideResNet28_10


def main():
    # 1. Parse Arguments (lr and label smoothing)
    parser = argparse.ArgumentParser(description="Run WideResNet28-10 training.")
    parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate')
    parser.add_argument('--ls', type=float, default=0.1, help='Label Smoothing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 2. Construct Configuration Dictionary
    # We explicitly define the fixed parameters required by the Trainer class
    base_name = f"WRN28-10_CIFAR10_lr={args.lr}_ls={args.ls}"
    
    config = {
        'base_name': base_name,
        'architecture': 'WideResnet28-10',
        'dataset': 'CIFAR10',
        'learning_rate': args.lr,
        'label_smoothing': args.ls

    }

    print(f"\n{'='*40}")
    print(f"STARTING EXPERIMENT: {base_name}")
    print(f"LR: {args.lr} | Label Smoothing: {args.ls}")
    print(f"{'='*40}\n")

    # 3. Load Data & Model
    train_loader, test_loader = get_dataloaders()
    
    # Initialize WideResNet28-10 for CIFAR-10 (10 classes)
    model = WideResNet28_10(num_classes=10).to(device)

    # 4. Initialize & Run Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
    )

    trainer.train_model()

    # 5. Cleanup
    del model, trainer, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    print("Training finished. Memory cleared.")

if __name__ == "__main__":
    main()