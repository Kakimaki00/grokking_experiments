import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttackLinf:
    def __init__(self, model):
        self.model = model
        self.num_steps = 10
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def kl_divergence(self, perturbed_outputs, base_outputs):
        p = F.softmax(base_outputs, dim=1)
        log_q = F.log_softmax(perturbed_outputs, dim=1)
        kl = F.kl_div(log_q, p, reduction='batchmean')
        return kl

    def perturb(self, inputs, epsilon=8/255, step_size=2/255, labels=None, loss_fn="kl"):
        # Ensure model is in eval mode (optional but recommended for attacks)
        # self.model.eval() 
        
        base_outputs = self.model(inputs)
        
        # Determine device from inputs
        device = inputs.device

        # Random initialization within the epsilon ball
        perturbed_inputs = inputs + 2 * epsilon * torch.rand_like(inputs) - epsilon

        # Clip to valid pixel range
        perturbed_inputs = torch.clamp(perturbed_inputs, min=0, max=1)

        # Initialize Adam buffers on the correct device
        grad_with_momentum = torch.zeros_like(inputs, device=device)
        grad_with_rms = torch.zeros_like(inputs, device=device)

        for t in range(self.num_steps):
            perturbed_inputs = perturbed_inputs.detach()
            perturbed_inputs.requires_grad_()

            # Get the outputs
            perturbed_outputs = self.model(perturbed_inputs)

            # Compute the selected metric
            if loss_fn == "cre":
                if labels is None:
                    raise ValueError("Labels required for CrossEntropy")
                distance = nn.CrossEntropyLoss()(perturbed_outputs, labels)
            elif loss_fn == "kl":
                distance = self.kl_divergence(perturbed_outputs, base_outputs)
            else:
                raise ValueError(f"Unknown loss function {loss_fn}")

            # Get gradient
            gradient = torch.autograd.grad(distance, perturbed_inputs)[0]

            # --- ADAM OPTIMIZER LOGIC ---
            # 1. Update moments
            grad_with_momentum = self.beta1 * grad_with_momentum + (1 - self.beta1) * gradient
            grad_with_rms = self.beta2 * grad_with_rms + (1 - self.beta2) * torch.pow(gradient, 2)

            # 2. Bias Correction (Must use t+1 to avoid div by zero)
            m_hat = grad_with_momentum / (1 - pow(self.beta1, t + 1))
            v_hat = grad_with_rms / (1 - pow(self.beta2, t + 1))

            # 3. Compute Update (Must use sqrt of second moment)
            updated_grad = m_hat / (torch.sqrt(v_hat) + self.eps)

            # Update inputs
            # Note: For Linf, we typically take the sign of the update. 
            # If using Adam, the update is already scaled, but standard PGD-Linf 
            # takes the sign of the final direction.
            perturbed_inputs = perturbed_inputs + step_size * torch.sign(updated_grad)

            # Project back to the epsilon ball
            perturbed_inputs = torch.min(torch.max(perturbed_inputs, inputs - epsilon), inputs + epsilon)

            # Clip to valid pixel range
            perturbed_inputs = torch.clamp(perturbed_inputs, min=0, max=1)

        return perturbed_inputs.detach()


class PGDAttackL2:
    def __init__(self, model, num_steps=10, random_start=True):
        self.model = model
        self.num_steps = num_steps
        self.random_start = random_start
        self.eps = 1e-12

    def _l2_normalize(self, x):
        # Per-sample normalization
        flat = x.view(x.size(0), -1)
        norm = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(self.eps)
        return (flat / norm).view_as(x)

    def _project_l2(self, x_adv, x, epsilon):
        # Project x_adv back into L2 ball around x
        delta = x_adv - x
        flat = delta.view(delta.size(0), -1)
        norm = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(self.eps)
        factor = torch.min(
            torch.ones_like(norm),
            epsilon / norm
        )
        delta = (flat * factor).view_as(delta)
        return x + delta

    def perturb(self, inputs, epsilon=128/255, step_size=32/255, labels=None, loss_fn="kl"):
        x = inputs.detach()
        device = x.device

        # Random start (uniform in L2 ball)
        if self.random_start:
            noise = torch.randn_like(x)
            noise = self._l2_normalize(noise)
            r = torch.rand(x.size(0), device=device)
            dim = x[0].numel()
            r = r.pow(1.0 / dim).view(-1, 1, 1, 1)
            x_adv = x + epsilon * r * noise
        else:
            x_adv = x.clone()

        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # Reference output for KL attack
        if loss_fn == "kl":
            with torch.no_grad():
                base_logits = self.model(x)

        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)

            if loss_fn == "cre":
                if labels is None:
                    raise ValueError("labels required for CE attack")
                loss = F.cross_entropy(logits, labels)

            elif loss_fn == "kl":
                loss = F.kl_div(
                    F.log_softmax(logits, dim=1),
                    F.softmax(base_logits, dim=1),
                    reduction="batchmean"
                )
            else:
                raise ValueError(f"Unknown loss_fn: {loss_fn}")

            grad = torch.autograd.grad(loss, x_adv)[0]

            # L2-normalized gradient ascent
            grad = self._l2_normalize(grad)
            x_adv = x_adv.detach() + step_size * grad

            # Projection + clamp
            x_adv = self._project_l2(x_adv, x, epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv.detach()
