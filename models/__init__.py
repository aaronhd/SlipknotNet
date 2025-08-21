def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'slipknotnet':
        from .slipknotnet import ResNet50UNet
        return ResNet50UNet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
