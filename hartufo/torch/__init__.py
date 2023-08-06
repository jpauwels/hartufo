from torch.utils.data._utils.collate import default_collate


def collate_dict_dataset(batch, features_key_name='features', target_key_name='target'):
    return [default_collate(x) for x in zip(*((d[features_key_name], d[target_key_name]) for d in batch))]
