import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from autoattack import AutoAttack
from models import WideResNet28_10
import argparse

def run_autoattack(model_path, num_samples, log_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-10 data
    transform = transforms.ToTensor()
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Extract subset to avoid massive computation time
    x_test = torch.stack([testset[i][0] for i in range(num_samples)])
    y_test = torch.tensor([testset[i][1] for i in range(num_samples)])

    model = WideResNet28_10(num_classes=10)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    # Handle both raw state_dicts and nested checkpoint dictionaries
    state_dict = checkpoint.get('model_state', checkpoint)
    
    # Strip "module." prefix if the model was saved using DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Initialize and run AutoAttack
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', log_path=log_file, device=device)
    adversary.run_standard_evaluation(x_test, y_test, bs=128)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to the .pth checkpoint")
    parser.add_argument('--samples', type=int, default=1000, help="Number of test samples to attack")
    parser.add_argument('--log', type=str, default='autoattack_results.log', help="Log file path")
    args = parser.parse_args()

    run_autoattack(args.model, args.samples, args.log)