import torch
from sklearn.metrics import f1_score
from TripletDataset import TripletDataset
from TripletDistanceMetric import TripletDistanceMetric

def evaluate(model, 
             model_name,
             test_data, 
             gpu_id, 
             batch_size: int=4, 
             distance_metric=TripletDistanceMetric.EUCLIDEAN,
             model_path: str=''):

    """
    Arguments:
     - model: the CLoSE model,                           type = torch Module
     - model_name: encoder name,                         type = string
                  'roberta-base', 'bert-base-uncased', or 'bert-base-cased'
     - test_data: test data,                             type = pandas dataframe
     - gpu_id: GPU id for assigning,                     type = int
     - batch_size: batch size,                           type = int
     - distance_metric: the metric for contrastive loss, type = func: Callable
     - model_path: path to the saved model,              type = string

    Returns:
     - the embeddings of anchor sentences,               type = numpy array
    """

    test = TripletDataset(test_data, model_name)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, drop_last=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device(gpu_id if use_cuda else "cpu")
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    if use_cuda:
        model = model.to(device)

    total_stance_acc_test = 0
    total_triplet_acc_test = 0
    embeddings = []
    pred_labels = []
    true_labels = []

    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            true_labels += test_label.tolist()
            test_label = test_label.to(device)
            masks = [] 
            input_ids = []
            for i in range(batch_size):
                for j in range(3):
                    masks.append(test_input[j]['attention_mask'][i])
                    input_ids.append(test_input[j]['input_ids'][i].squeeze(1))
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

            for i in range(dist_pos.size()[0]):
                if dist_pos[i] < dist_neg[i]:
                    total_triplet_acc_test += 1

            # Stance prediction
            pred_labels += final_layer.argmax(dim=1).detach().cpu().tolist()
            stance_acc = (final_layer.argmax(dim=1) == test_label).sum().item()
            total_stance_acc_test += stance_acc
    
            if embeddings == []:
                embeddings = output_anchor
            else:
                embeddings = torch.cat((embeddings, output_anchor))

    f1 = f1_score(true_labels, pred_labels)
    print(f'Test Stance f1: {f1: .3f}, Test Stance Accuracy: {total_stance_acc_test / len(test_data): .3f} | Test Triplet Accuracy: {total_triplet_acc_test / len(test_data): .3f}')
    
    return embeddings.cpu().detach().numpy()
