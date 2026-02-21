import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from utils import PGDAttackLinf, PGDAttackL2

class Trainer:
    def __init__(self, model, train_loader, test_loader, config):
        self.device = "cuda"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
        # Setup Hyperparameters
        self.epochs = 10000000
        self.label_smoothing = config["label_smoothing"]

        # Attacks for Logging (Linf and L2)
        # Assuming PGDAttack classes accept model + config parameters
        self.attack_linf = PGDAttackLinf(self.model)
        self.attack_l2 = PGDAttackL2(self.model)
        
        # Directory setup
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.save_dir = os.path.join("training_results", f"{config["base_name"]}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Optimizer
        lr = self.config["lr"]
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        # Scheduler (Assuming StepLR if not specified, usually needed)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

        # Loss Function
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def _create_smoothed_targets(self, labels, num_classes):
        smoothing = self.config["smoothing"]
        batch_size = labels.size(0)
        one_hot = torch.zeros(batch_size, num_classes, device=self.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        smoothed = (1.0 - smoothing) * one_hot + smoothing / num_classes
        return smoothed

    def compute_loss(self, outputs, labels):
        log_probs = F.log_softmax(outputs, dim=1)
        num_classes = outputs.size(1)
        targets = self._create_smoothed_targets(labels, num_classes, self.label_smoothing)
        loss = self.criterion(log_probs, targets)
        return loss

    def _evaluate_batch(self, inputs, labels, attack=None):
        """Helper to compute loss/acc for a specific input type (clean or adv)"""
        if attack:
            # Generate attacks in eval mode to avoid batchnorm updates during generation
            self.model.eval()
            inputs = attack.perturb(inputs, labels=labels)
            self.model.train()

        # Compute metrics
        outputs = self.model(inputs)
        loss = self.compute_loss(outputs, labels)
        _, pred = outputs.max(1)
        correct = pred.eq(labels).sum().item()
        
        return loss.item(), correct

    def train_one_epoch(self):
        self.model.train()
        
        metrics = {
            'clean_loss': 0, 'clean_acc': 0,
            'linf_loss': 0, 'linf_acc': 0,
            'l2_loss': 0, 'l2_acc': 0,
            'total': 0
        }

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            metrics['total'] += labels.size(0)

            # 1. Standard Training (Clean Only)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Record Clean Metrics
            metrics['clean_loss'] += loss.item()
            _, pred = outputs.max(1)
            metrics['clean_acc'] += pred.eq(labels).sum().item()

            # 2. Logging Adversarial Metrics (No Training)
            # We assume model is in train mode, but we use eval inside helper for generation
            linf_loss, linf_corr = self._evaluate_batch(inputs, labels, self.attack_linf)
            metrics['linf_loss'] += linf_loss
            metrics['linf_acc'] += linf_corr

            l2_loss, l2_corr = self._evaluate_batch(inputs, labels, self.attack_l2)
            metrics['l2_loss'] += l2_loss
            metrics['l2_acc'] += l2_corr

        # Average metrics
        n = len(self.train_loader)
        total = metrics['total']
        
        return {k: (v / n if 'loss' in k else 100. * v / total) for k, v in metrics.items() if k != 'total'}

    def test(self):
        self.model.eval()
        
        metrics = {
            'clean_loss': 0, 'clean_acc': 0,
            'linf_loss': 0, 'linf_acc': 0,
            'l2_loss': 0, 'l2_acc': 0,
            'total': 0
        }
        
        val_criterion = nn.CrossEntropyLoss()

        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            metrics['total'] += labels.size(0)

            # Clean
            with torch.no_grad():
                outputs = self.model(inputs)
                metrics['clean_loss'] += val_criterion(outputs, labels).item()
                metrics['clean_acc'] += outputs.max(1)[1].eq(labels).sum().item()

            # Linf Evaluation
            adv_linf = self.attack_linf.perturb(inputs, labels=labels)
            with torch.no_grad():
                out_linf = self.model(adv_linf)
                metrics['linf_loss'] += val_criterion(out_linf, labels).item()
                metrics['linf_acc'] += out_linf.max(1)[1].eq(labels).sum().item()

            # L2 Evaluation
            adv_l2 = self.attack_l2.perturb(inputs, labels=labels)
            with torch.no_grad():
                out_l2 = self.model(adv_l2)
                metrics['l2_loss'] += val_criterion(out_l2, labels).item()
                metrics['l2_acc'] += out_l2.max(1)[1].eq(labels).sum().item()

        n = len(self.test_loader)
        total = metrics['total']

        return {k: (v / n if 'loss' in k else 100. * v / total) for k, v in metrics.items() if k != 'total'}

    def train_model(self):
        print(f"Starting standard training (with adv logging) on {self.device}...")

        for epoch in range(self.epochs):
            train_metrics = self.train_one_epoch()
            test_metrics = self.test()
            self.scheduler.step()

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            print(f"  Train: Clean {train_metrics['clean_acc']:.1f}% | Linf {train_metrics['linf_acc']:.1f}% | L2 {train_metrics['l2_acc']:.1f}%")
            print(f"  Test : Clean {test_metrics['clean_acc']:.1f}% | Linf {test_metrics['linf_acc']:.1f}% | L2 {test_metrics['l2_acc']:.1f}%")

            self.save_checkpoint(epoch, train_metrics, test_metrics)

    def save_checkpoint(self, epoch, train_metrics, test_metrics):
        log_path = os.path.join(self.save_dir, "training_log.json")
        
        log_entry = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'test': test_metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f: history = json.load(f)
        else:
            history = []
            
        history.append(log_entry)
        
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)

        if (epoch + 1) % 100 == 0:
            state = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': self.config,
                'metrics': log_entry
            }
            path = os.path.join(self.save_dir, f"checkpoint_ep{epoch+1}.pth")
            torch.save(state, path)
            print(f"Saved checkpoint: {path}")
