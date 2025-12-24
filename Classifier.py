import torch
from autoencoders.noise_layer import AdditiveGaussianNoise

sigma = 1e-3  # Noise-to-signal ratio. From Information-v3 repo

class MNIST_Classifier(torch.nn.Module):
    """
    This is the classifer from Omnigrok paper
    from MNIST experiments
    """

    def __init__(
            self,
            width,
            activation_fn,
            # for MI estiamtion
            sigma: float = 0.1,
            relative_scale=True,
    ):
        super().__init__()
        self.sigma = sigma

        # Noise.
        self.agn = AdditiveGaussianNoise(self.sigma, enabled_on_inference=True, relative_scale=relative_scale)

        # Activations.
        self.activation = activation_fn()

        # Dense.
        self.linear_1 = torch.nn.Linear(784, width)
        self.linear_2 = torch.nn.Linear(width, width)
        self.linear_3 = torch.nn.Linear(width, 10)

    def forward(self, x: torch.tensor,
                all_layers: bool = False,
                apply_agn: bool = True
                ) -> torch.tensor:
        """

        :param x: Input image tensor, not flatten.
        :param all_layers: Return dict of outputs of all layers.
        :param apply_agn: Whether to apply AGN layer to activations. Turn off for dead_nrns computations.
        """
        x = (torch.flatten(x, 1))
        # Dense №1
        layer_1 = self.activation(self.linear_1(x))
        if apply_agn:
            layer_1 = self.agn(layer_1)
        # Dense №2
        layer_2 = self.activation(self.linear_2(layer_1))
        if apply_agn:
            layer_2 = self.agn(layer_2)
        # Dense №3
        layer_3 = self.linear_3(layer_2)
        if apply_agn:
            layer_3 = self.agn(layer_3)

        if all_layers:
            return {
                'layer 1': layer_1,
                'layer 2': layer_2,
                'layer 3': layer_3,
            }
        else:
            return layer_3


# class MNIST_Classifier_mlp4(torch.nn.Module):
#     """
#     This is the classifer from Omnigrok paper
#     from MNIST experiments
#     """
#
#     def __init__(
#             self,
#             width,
#             activation_fn,
#             sigma: float = 0.1
#     ):
#         super().__init__()
#         self.sigma = sigma
#
#         # Noise.
#         self.agn = AdditiveGaussianNoise(sigma, enabled_on_inference=True)
#
#         # Activations.
#         self.activation = activation_fn()
#
#         # Dense.
#         self.linear_1 = torch.nn.Linear(784, width)
#         self.linear_2 = torch.nn.Linear(width, width)
#         self.linear_3 = torch.nn.Linear(width, width)
#         self.linear_4 = torch.nn.Linear(width, 10)
#
#     def forward(self, x: torch.tensor, all_layers: bool = False) -> torch.tensor:
#         # Dense №1
#         x = self.agn(torch.flatten(x, 1))
#
#         x = self.linear_1(x)
#         layer_1 = self.activation(x)
#
#         # Dense №2
#         x = self.agn(layer_1)
#         x = self.linear_2(x)
#         layer_2 = self.activation(x)
#
#         # Dense №3
#         x = self.agn(layer_2)
#         x = self.linear_2(x)
#         layer_3 = self.activation(x)
#
#         # Dense №4
#         x = self.agn(layer_3)
#         layer_4 = self.linear_4(x)
#
#         if all_layers:
#             return {
#                 'layer 1': layer_1,
#                 'layer 2': layer_2,
#                 'layer 3': layer_3,
#                 'layer 4': layer_4,
#             }
#         else:
#             return layer_4