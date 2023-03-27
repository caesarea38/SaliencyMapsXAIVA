import warnings
import torch
from torch.optim import SGD 
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

def train(
    model, 
    train_loader, 
    val_loader, 
    test_loader,
    current_epoch, 
    metrics_storage, 
    args,
    session_state
    ):

    optimizer = SGD(
        model.parameters(),
        lr=args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # to store metrics calculated for training as well as validation sets
    model.train()
    total_mini_batches_train = len(train_loader)

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        images, class_labels, uq_idxs = batch
        images = images.to(args.device)
        class_labels = class_labels.to(args.device)

        # Extract features and output after linear classifier with model
        out = model(images)
        loss = torch.nn.CrossEntropyLoss()(out, class_labels)
        
        # Train acc
        _, preds = out.max(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx != total_mini_batches_train-1:
            metrics_storage.update(mode='training', loss=loss.item(), size=class_labels.size(0), preds=preds.tolist(), labels=class_labels.tolist(), last_batch = False)
        else: 
            metrics_storage.update(mode='training', loss=loss.item(), size=class_labels.size(0), preds=preds.tolist(), labels=class_labels.tolist(), last_batch = True)
            print(metrics_storage.metrics["train_loss"].all_values_avg)
        # TODO: ONLY FOR DEBUG
        #metrics_storage.update(mode='training', loss=loss.item(), size=class_labels.size(0), preds=preds.tolist(), labels=class_labels.tolist(), last_batch = True)
        #if batch_idx == 0:
        #    break
    
    print(f'Train Epoch: {current_epoch} Avg Loss: {metrics_storage.metrics["train_loss"].avg} | Acc: {metrics_storage.metrics["train_acc"].avg}')
        
    # Evaluate on the validation set
    print('Evaluating on the disjoint validation set...')
    model.eval()

    total_mini_batches_val = len(val_loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            images, class_labels, uq_idxs = batch
            images = images.to(args.device)
            class_labels = class_labels.to(args.device)

            # Extract features and output after linear classifier with model
            out = model(images)

            # Train acc
            _, preds = out.max(1)

            loss = torch.nn.CrossEntropyLoss()(out, class_labels)

            if batch_idx != total_mini_batches_val-1:
                metrics_storage.update(mode='validation', loss=loss.item(), size=class_labels.size(0), preds=preds.tolist(), labels=class_labels.tolist(), last_batch = False)
            else: 
                metrics_storage.update(mode='validation', loss=loss.item(), size=class_labels.size(0), preds=preds.tolist(), labels=class_labels.tolist(), last_batch = True)

            # TODO: ONLY FOR DEBUG
            #metrics_storage.update(mode='validation', loss=loss.item(), size=class_labels.size(0), preds=preds.tolist(), labels=class_labels.tolist(), last_batch = True)

            #if batch_idx == 0:
            #    break

        print(f'Val Avg Loss: {metrics_storage.metrics["val_loss"].avg} | Acc: {metrics_storage.metrics["val_acc"].avg}')

    preds_probs, labels, preds = test(
        model=model,
        test_loader=test_loader,
        args=args
    )

    session_state['training_monitoring']['predictions'][current_epoch] = (preds_probs, preds)
    session_state['training_monitoring']['labels'][current_epoch] = labels

def test(model, test_loader, args):
    model.eval()
    pred_probs = []
    labels = []
    preds = []

    print('Evaluating on the disjoint test set...')
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images, class_labels, uq_idxs = batch
            images = images.to(args.device)
            class_labels = class_labels.to(args.device)

            # Extract features and output after linear classifier with model
            out = model(images)
            _, pred = out.max(1)
            preds += pred.tolist()
            pred_probs += out.tolist()
            labels += class_labels.tolist()
    return pred_probs, labels, preds