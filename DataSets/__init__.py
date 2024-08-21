from .testa import TestA

dataset_list = {
    "TestSetA": TestA
}


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)
