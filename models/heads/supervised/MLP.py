import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_size, configs, seed):
        super(MLP, self).__init__()
        self.logits = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        self.loss = configs['loss']
        self.optimizer = configs['optimizer'](self.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
        self.scheduler = configs['scheduler'](self.optimizer, **configs['sched_params'])

    def forward(self, x):
        logits = self.logits(x)
        return self.logits(x).squeeze(1) if len(logits.shape) == 3 else logits  
    
    def to_train(self, train_loader, loss, opt, sched, device):
        self.to(device)
        self.train() 
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = self.forward(x)

            loss_value = loss(logits, y)
            loss_value.backward()
            opt.step()

            running_loss += loss_value.item() 
            probs = F.softmax(logits, dim=1) 
            argmax = torch.argmax(probs, dim=1)
            total_correct += (argmax == y).sum().item()
            total_samples += x.size(0)

        avg_loss = running_loss / total_samples
        accuracy = total_correct / total_samples
        sched.step(running_loss / len(train_loader))
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, val_loader, loss, device):
        self.to(device)
        self.eval() 
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            logits = self.forward(x)
            loss_value = loss(logits, y)

            total_loss += loss_value.item() * x.size(0)
            probs = F.softmax(logits, dim=-1) 
            argmax = torch.argmax(probs, dim=1)
            total_samples += x.size(0)

            all_preds.extend(argmax.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg_val_loss = total_loss / total_samples
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return  avg_val_loss, acc, f1


    def run(self, train_loader, val_loader, epochs, device, run_idx=0, path=''):
        min_val_loss = float('inf')
        val_loss_epochs = []

        for epoch in range(epochs):
            train_loss, train_acc = self.to_train(train_loader, self.loss, self.optimizer, self.scheduler, device)
            val_loss, val_acc, val_f1 = self.evaluate(val_loader, self.loss, device)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.state_dict(), f'{path}/{self.__class__.__name__}_best_model_run_{run_idx}.pth')

            val_loss_epochs.append(val_loss)

            print(f'Run {run_idx+1}/Epoch {epoch+1}/{epochs}, '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            wandb.log({
                f"train_loss/run_{run_idx}": train_loss,
                f"val_loss/run_{run_idx}": val_loss,
                f"val_acc/run_{run_idx}": val_acc,
                f"val_f1/run_{run_idx}": val_f1,
                f"step/run_{run_idx}": epoch
            })
        print(f'{path}/validation_loss_run_{run_idx}.pth')
        print(f'{path}/{self.__class__.__name__}_best_model_run_{run_idx}.pth')
        torch.save(torch.tensor(val_loss_epochs), f'{path}/validation_loss_run_{run_idx}.pth')
        self.load_state_dict(torch.load(f'{path}/{self.__class__.__name__}_best_model_run_{run_idx}.pth'))
