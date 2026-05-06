# =========================================== #
#  COUNTERFACTUAL PGD ATTACK #
# =========================================== #

# +
import torch

from datasets.loader import flip_sensitive

def counterfactual_loss(model, x, y, s_batch, s_index, lambd = 0.5):
    
    model.eval()
    
    device = next(model.parameters()).device

    x = x.to(device)
    y = y.to(device)
    s_batch = s_batch.to(device)
    
    s_cf = 1 - s_batch
    
    x_a = x.clone()
    x_b = x.clone()

    x_a[:, s_index] = s_batch.squeeze()
    x_b[:, s_index] = s_cf.squeeze()

    out_a = model(x_a).view(-1)
    out_b = model(x_b).view(-1)

    loss_a = hinge_loss_pgd(out_a, y, C=1.0)
    loss_b = hinge_loss_pgd(out_b, y, C=1.0)
    
    loss = torch.abs(lambd*loss_a - (1-lambd)*loss_b)
    
    return loss


# -

def hinge_loss_pgd(outputs, targets, C=1.0):

    return torch.clamp(1 - targets * outputs, min=0)  # <-- NO mean


def pgd_counterfactual_attack(
    model, x, y, s_batch, s_index,
    lambd=0.5, eps=0.1, alpha=0.01, steps=40,
    p=float('inf'),
    target_group=None
):

    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    
    device = next(model.parameters()).device

    x = x.to(device)
    y = y.to(device)
    s_batch = s_batch.to(device)

    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    if target_group is not None:
        mask = (s_batch == target_group)
    else:
        mask = torch.ones_like(s_batch, dtype=torch.bool)

    for _ in range(steps):

        loss =  counterfactual_loss(model, x_adv, y, s_batch, s_index, lambd)

        loss = loss[mask].mean()
        loss.backward()

        with torch.no_grad():

            grad = x_adv.grad

            grad[~mask] = 0.0

            if p == float('inf'):
                step = alpha * grad.sign()
            elif p == 2:
                grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1, keepdim=True)
                step = alpha * grad / (grad_norm + 1e-8)

            x_adv = x_adv + step

            delta = x_adv - x

            if p == float('inf'):
                delta = torch.clamp(delta, -eps, eps)

            x_adv = x + delta

        x_adv = x_adv.detach()
        x_adv.requires_grad = True

    return x_adv.detach()

def generate_attacked_data(model, X, y, s, s_index, eps, lambd, target_group=None):

    device = next(model.parameters()).device

    X = X.to(device)
    y = y.to(device)
    s = s.to(device)

    X_adv = pgd_counterfactual_attack(
        model, X, y, s,
        s_index=s_index,
        lambd=lambd,
        eps=eps,
        alpha=0.01,
        steps=100,
        p=float('inf'),
        target_group=target_group  
    )

    X_cf = flip_sensitive(X_adv, s_index)

    return X_adv, X_cf
