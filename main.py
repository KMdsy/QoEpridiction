import argparse, torch, time
from data_loader import load_data, evaluate
from models import GAT
from configs import train_configs
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def eval(model, data, scaler):
    model.eval()
    preds = []
    labels = []
    data_size = data['features'].shape[0]
    for step in range(data_size):
        batch = {k: v[step].to(device) for k, v in data.items()}
        output = model(**batch)
        preds.append(output.pred.cpu().detach().numpy())
        labels.append(data['labels'][step].cpu().detach().numpy())
    preds = np.array(preds)
    labels = np.array(labels)
    metrics = evaluate(preds, labels, scaler)
    return metrics


if __name__ == '__main__':
    data = load_data()
    # Dataset
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    # pop scaler
    train_scaler, val_scaler, test_scaler = train_data.pop('scaler'), val_data.pop('scaler'), test_data.pop('scaler')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = GAT(nfeat=train_configs.n_features,
                npred=train_configs.n_pred,
                nhid=args.hidden, # TODO
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
    model.to(device)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)
    
    # train
    train_size = train_data['features'].shape[0]
    pbar = tqdm(range(args.epochs * train_size))
    best_performance = -np.inf
    best_model = None
    counter = 0 # early stop counter
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        # train on minibatch
        for step in range(train_size):
            batch = {k: v[step].to(device) for k, v in train_data.items()}
            output = model(**batch)
            loss = output.loss

            loss.backward()
            optimizer.step()
            pbar.update(1)

        if epoch % 10 == 0:
            # eval
            model.eval()
            val_metrics = eval(model, val_data, val_scaler)

            if val_metrics['rmse'] > best_performance:
                best_performance = val_metrics['rmse']
                counter = 0
                # test on best model
                test_metrics = eval(model, test_data, test_scaler)
                # save best model to file
                best_model = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics
                }
                torch.save(best_model, 'best_model.pt')
            else:
                counter += 1
                if counter == args.patience:
                    print('Early stop!')
                    break
