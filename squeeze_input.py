import torchvision.models as models


def squeeze_weights(module):
    module.weight.data = module.weight.data.sum(dim=1)[:, None]
    module.in_channels = 1


if __name__ == '__main__':
    mobilenet_v2 = models.mobilenet_v2()
    # смотрим, как называется первый сверточный слой
    print(mobilenet_v2.features[0])
    # видим ConvNormActivation c входной размерностью 3:
    # Sequential(
    # (0): ConvNormActivation(
    #   (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # ...
    # то есть нам нужно взять features[0][0]
    squeeze_weights(mobilenet_v2.features[0][0])
    print('\n\nNEW INPUT\n')
    print(mobilenet_v2.features[0])
    # теперь видим одноканальный вход
    # Sequential(
    #     (0): ConvNormActivation(
    #     (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
