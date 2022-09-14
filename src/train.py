import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
from TripletDataset import TripletDataset
from TripletDistanceMetric import TripletDistanceMetric


def compute_loss(data_input, 
                label,
                model,
                criterion, 
                batch_size, 
                device, 
                distance_metric,
                triplet_margin,
                alpha, 
                total_triplet_acc, 
                total_stance_acc):
    label = label.to(device)
    masks = [] 
    input_ids = []
    for i in range(batch_size):
        for j in range(3):
            masks.append(data_input[j]['attention_mask'][i])
            input_ids.append(data_input[j]['input_ids'][i].squeeze(1))
    masks = torch.cat(masks, dim=0)
    input_ids = torch.cat(input_ids, dim=0)

    masks = masks.to(device)  
    input_ids = input_ids.to(device)  

    sentence_embeddings, final_layer = model(input_ids, masks, device, batch_size)

    indices = torch.tensor([3*i for i in range(batch_size)]).to(device)
    output_anchor = torch.index_select(sentence_embeddings, 0, indices)
    output_pos = torch.index_select(sentence_embeddings, 0, indices+1)
    output_neg = torch.index_select(sentence_embeddings, 0, indices+2)
    
    # Triplet contrastive loss
    dist_pos = distance_metric(output_anchor, output_pos)
    dist_neg = distance_metric(output_anchor, output_neg)
    dist_loss = F.relu(dist_pos - dist_neg + triplet_margin)
    dist_loss = dist_loss.mean()

    for i in range(batch_size):
        if dist_pos[i] < dist_neg[i]:
            total_triplet_acc += 1

    # Stance prediction
    batch_loss = criterion(final_layer, label)

    total_loss = (1 - alpha) * dist_loss + alpha * batch_loss
    #total_loss_train += total_loss.item()
    
    stance_acc = (final_layer.argmax(dim=1) == label).sum().item()
    total_stance_acc += stance_acc

    return total_loss, total_triplet_acc, total_stance_acc


def train(model, 
          model_name,
          train_data, 
          val_data, 
          learning_rate, 
          epochs, 
          gpu_id, 
          batch_size: int=4, 
          distance_metric=TripletDistanceMetric.EUCLIDEAN,
          triplet_margin: float=1,
          alpha: float=0.5,
          model_path: str=''):

    """
    Arguments:
     - model: the CLoSE model,                           type = torch Module
     - model_name: encoder name,                         type = string
                  'roberta-base', 'bert-base-uncased', or 'bert-base-cased'
     - train_data: train data,                           type = pandas dataframe
     - val_data: valid data,                             type = pandas dataframe
     - learning_rate: learning rate,                     type = float
     - epochs: epcohs,                                   type = int
     - gpu_id: GPU id for assigning,                     type = int
     - batch_size: batch size,                           type = int
     - distance_metric: the metric for contrastive loss, type = func: Callable
     - triplet_margin: the epsilon in contrastive loss,  type = float
     - alpha: alpha in the final loss,                   type = float
     - model_path: path for saving the trained model,    type = string
    """

    train, val = TripletDataset(train_data, model_name), TripletDataset(val_data, model_name)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size, drop_last=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device(gpu_id if use_cuda else "cpu")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = learning_rate)

    best_loss = 100

    if use_cuda:
            model = model.to(device)
            criterion = criterion.to(device)

    for epoch_num in range(epochs):

            total_stance_acc_train = 0
            total_triplet_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                total_loss, total_triplet_acc_train, total_stance_acc_train = compute_loss(train_input, 
                                                                                    train_label, 
                                                                                    model,
                                                                                    criterion,
                                                                                    batch_size, 
                                                                                    device, 
                                                                                    distance_metric,
                                                                                    triplet_margin,
                                                                                    alpha, 
                                                                                    total_triplet_acc_train, 
                                                                                    total_stance_acc_train)
                total_loss_train += total_loss.item()

                model.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            total_stance_acc_val = 0
            total_triplet_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:
                    total_loss, total_triplet_acc_val, total_stance_acc_val = compute_loss(val_input, 
                                                                                    val_label, 
                                                                                    model,
                                                                                    criterion,
                                                                                    batch_size, 
                                                                                    device, 
                                                                                    distance_metric, 
                                                                                    triplet_margin,
                                                                                    alpha,
                                                                                    total_triplet_acc_val, 
                                                                                    total_stance_acc_val)
                    total_loss_val += total_loss.item()

            # Save model
            total_loss_val = total_loss_val / len(val_data)
            if total_loss_val < best_loss: 
                best_loss = total_loss_val
                print("save all model to {}".format(model_path))
                output = open(model_path, mode="wb")
                torch.save(model.state_dict(), output)

            print(f'Epochs: {epoch_num + 1} ')
            print(f'\t| Train Loss: {total_loss_train / len(train_data): .3f} | Train Stance Accuracy: {total_stance_acc_train / len(train_data): .3f} | Train Triplet Accuracy: {total_triplet_acc_train / len(train_data): .3f}')
            print(f'\t| Val Loss: {total_loss_val: .3f} | Val Stance Accuracy: {total_stance_acc_val / len(val_data): .3f} | Val Triplet Accuracy: {total_triplet_acc_val / len(val_data): .3f}')

