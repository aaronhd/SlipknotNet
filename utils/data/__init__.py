def get_dataset(dataset_name):
    if dataset_name == 'suture_line':
        from .suture_data import ImageMaskDataset
        return ImageMaskDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))
        