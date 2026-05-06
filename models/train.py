import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .mlp import build_model
from datasets.loader import flip_sensitive

def hinge_loss(output, target, C=1.0):
    
    loss = C*torch.mean(torch.clamp(1 - output.squeeze() * target, min=0))
    
    return loss

def train_model(model, dataset,
                epochs=50,
                batch_size=64,
                lr=1e-3,
                C=1.0,
                device="cuda:0"):

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(dataset.X_train, dataset.y_train),
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(epochs):

        model.train()

        epoch_loss = 0

        for xb,yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            out = model(xb)

            loss = hinge_loss(out,yb,C)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()*xb.size(0)

        epoch_loss /= len(train_loader.dataset)

        if (epoch+1)%5==0:

            print(f"Epoch {epoch+1}/{epochs} | Loss {epoch_loss:.4f}")

    print("\nTraining finished\n")
    
    model.eval()
    
    X_train_cf = flip_sensitive(dataset.X_train, dataset.s_index)
    X_test_cf = flip_sensitive(dataset.X_test, dataset.s_index)
    
    with torch.no_grad():

        y_pred_train = torch.sign(model(dataset.X_train.to(device)))
        y_pred_test = torch.sign(model(dataset.X_test.to(device)))
        
        y_pred_train_cf = torch.sign(model(X_train_cf.to(device)))
        y_pred_test_cf = torch.sign(model(X_test_cf.to(device)))

    train_acc = accuracy(y_pred_train, dataset.y_train)
    test_acc  = accuracy(y_pred_test ,dataset.y_test)
    
    te_count_train, te_perc_train = TE(y_pred_train, y_pred_train_cf)
    te_count_test, te_perc_test = TE(y_pred_test, y_pred_test_cf)
    
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test  Accuracy: {test_acc:.3f}")
    
    print("\n\n")
    
    print(f"Train Total Effect: {te_count_train} ({te_perc_train:.2f}%)")
    print(f"Test Total Effect: {te_count_test} ({te_perc_test:.2f}%)")

    return model

def accuracy(y_pred, y):

    acc = (y_pred.cpu() == y).float().mean()

    return acc.item()

def TE(y_pred, y_pred_cf):
   
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_pred_cf, torch.Tensor):
        y_pred_cf = y_pred_cf.detach().cpu().numpy()

    diff_mask = (y_pred != y_pred_cf)
    
    te_count = diff_mask.sum()
    te_perc = diff_mask.mean() * 100
    
    return te_count, te_perc
