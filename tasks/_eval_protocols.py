from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score,f1_score,recall_score
from sklearn.preprocessing import label_binarize
import torch.utils.data as Data
from models import *
from tqdm import tqdm

class Linear_probe(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.linear = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.linear(x)


def fit_mlp(features, y, test_features, test_y):

    samples_num = features.shape[0]
    torch_dataset_train = Data.TensorDataset(torch.tensor(features), torch.tensor(y))
    # torch_dataset_val = Data.TensorDataset(torch.tensor(features), torch.tensor(y))
    torch_dataset_test = Data.TensorDataset(torch.tensor(test_features), torch.tensor(test_y))

    loader_train = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=256,
        shuffle=True,
    )
    # loader_val = Data.DataLoader(
    #     dataset=torch_dataset_val,
    #     batch_size=256,
    #     shuffle=True,
    # )
    loader_test = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=256,
        shuffle=False,
    )

    mlp_model = Linear_probe(features.shape[1], int(y.max()+1))
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    val_loss_epoch = []
    test_loss_epoch = []
    acc_epoch = []
    roc_epoch = []
    f1_epoch = []
    recall_epoch = []
    for epoch in tqdm(range(100)):
        train_loss = []
        val_loss = []
        for step, (batch_x, batch_y) in enumerate(loader_train):
            optimizer.zero_grad()
            output = mlp_model(batch_x)
            loss = loss_func(output, batch_y)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        # with torch.no_grad():
        #     for step, (batch_x, batch_y) in enumerate(loader_val):
        #         output = mlp_model(batch_x)
        #         loss = loss_func(output, batch_y)
        #         val_loss.append(loss.item())

        test_pred = []
        test_loss = []
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(loader_test):
                output = mlp_model(batch_x)
                loss = loss_func(output, batch_y)
                test_loss.append(loss.item())
                test_pred.append(output)
        test_pred = torch.cat(test_pred, dim=0)
        test_pred = torch.functional.F.softmax(test_pred, dim=1)
        test_pred_argmax = torch.argmax(test_pred, dim=1)
        acc = accuracy_score(test_y, test_pred_argmax.numpy())
        
        test_labels_onehot = label_binarize(test_y, classes=np.arange(y.max() + 1))
        if int(y.max()+1) >2:
            #print(test_pred.numpy())
            #mean_val = np.nanmean(test_pred.numpy())
            #test_pred_filled = np.where(np.isnan(test_pred.numpy()), mean_val, test_pred.numpy())
            roc = roc_auc_score(test_y, test_pred.numpy(), multi_class='ovr')
        else:
            roc = roc_auc_score(test_y, test_pred.numpy()[:, 1])
        # val_loss_epoch.append(np.mean(np.mean(val_loss)))
        #print(np.mean(np.array(test_loss)))
        test_loss_epoch.append(np.mean(np.array(test_loss)))
        acc_epoch.append(acc)
        roc_epoch.append(roc)

        f1s = f1_score(test_y,  test_pred_argmax.numpy(),average="macro")
        recalls = recall_score(test_y,  test_pred_argmax.numpy(),average="macro")
        f1_epoch.append(f1s)
        recall_epoch.append(recalls)


    # val_loss_epoch = np.array(val_loss_epoch)
    min_idx = np.argmin(test_loss_epoch)
    return test_pred, { 'acc': acc_epoch[min_idx]
                       , 'auroc': roc_epoch[min_idx]
                       ,'f1_score': f1_epoch[min_idx]
                       ,'recall':recall_epoch[min_idx]
                       }


