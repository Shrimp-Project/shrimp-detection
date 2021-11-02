import torch

from datasets.shrimp_dataset import ShrimpDataset

def load_data(dataset_detail):
    dataset = ShrimpDataset(dataset_detail[0], dataset_detail[1], False)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_detail[2], shuffle=True, collate_fn=dataset.collate_fn)

    return dataset_loader