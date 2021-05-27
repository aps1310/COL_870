# Imports
import torch
from torch.nn import Parameter, Module


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9, train=True):
    """
    Args:
        X: Input Tensor
        gamma: Learned parameter
        beta: Learned parameter
        moving_mean: Used in test mode
        moving_var: Used in test mode.
        eps: Calculation Stabilizing Constant
        momentum: It will decide how we update moving_mean and moving_var
        train: Indicates whether training or testing
    """
    if not train:
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data


class MyBatchNorm(Module):
    """
    Args:
         num_filters: Number of channels
    """
    def __init__(self, num_filters):
        super().__init__()
        shape = (1, num_filters, 1, 1)

        self.gamma = Parameter(torch.ones(shape))
        self.beta = Parameter(torch.zeros(shape))

        self.register_buffer('moving_mean', torch.zeros(shape), persistent=True)
        self.register_buffer('moving_var', torch.ones(shape), persistent=True)



    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = batch_norm(
            x, self.gamma, self.beta, self.moving_mean, self.moving_var, train=self.training)
        return y


def instance_norm(X, gamma, beta, eps=1e-5, train=True):
    """
    Args:
        X: Input Tensor
        gamma: Learned parameter
        beta: Learned parameter
        eps: Calculation Stabilizing Constant
        train: Indicates whether training or testing
    """
    mean = X.mean(dim=(2, 3), keepdim=True)
    var = ((X - mean) ** 2).mean(dim=(2, 3), keepdim=True)
    X_hat = (X - mean) / torch.sqrt(var + eps)

    Y = gamma[:X.shape[0], :, :, :] * X_hat + beta[:X.shape[0], :, :, :]
    return Y


class MyInstanceNorm(Module):
    """
    Args:
         num_filters: Number of channels
    """

    def __init__(self, num_filters, batch_size=128):
        super().__init__()
        shape = (batch_size, num_filters, 1, 1)

        self.gamma = Parameter(torch.ones(shape))
        self.beta = Parameter(torch.zeros(shape))

    def forward(self, x):
        y = instance_norm(x, self.gamma, self.beta, train=self.training)
        return y


'''  Dropping Last Batch 
def layer_norm(X, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9, train=True):
    """
    Args:
        X: Input Tensor
        gamma: Learned parameter
        beta: Learned parameter
        moving_mean: Used in test mode
        moving_var: Used in test mode.
        eps: Calculation Stabilizing Constant
        momentum: It will decide how we update moving_mean and moving_var
        train: Indicates whether training or testing
    """
    if not train:
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean[:X.shape[0], :, :, :]) / torch.sqrt(moving_var[:X.shape[0], :, :, :] + eps)
    else:
        mean = X.mean(dim=(1,2,3), keepdim=True)
        #print("mean",mean.shape);
        #print("moving_mean",moving_mean.shape);
        var = ((X - mean) ** 2).mean(dim=(1,2,3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean[:X.shape[0], :, :, :] = momentum * moving_mean[:X.shape[0], :, :, :] + (1.0 - momentum) * mean
        moving_var[:X.shape[0], :, :, :] = momentum * moving_var[:X.shape[0], :, :, :] + (1.0 - momentum) * var
    Y = gamma[:X.shape[0], :, :, :] * X_hat + beta[:X.shape[0], :, :, :]
    return Y, moving_mean.data, moving_var.data


class MyLayerNorm(Module):
    """
    Args:
         num_filters: Number of channels
    """

    def __init__(self, num_filters, batch_size=128):
        super().__init__()
        shape = (batch_size, 1, 1, 1)

        self.gamma = Parameter(torch.ones(shape))
        self.beta = Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = layer_norm(
            x, self.gamma, self.beta, self.moving_mean, self.moving_var, train=self.training)
        return y
'''


def layer_norm(X, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9, train=True):
    """
    Args:
        X: Input Tensor
        gamma: Learned parameter
        beta: Learned parameter
        moving_mean: Used in test mode
        moving_var: Used in test mode.
        eps: Calculation Stabilizing Constant
        momentum: It will decide how we update moving_mean and moving_var
        train: Indicates whether training or testing
    """
    if not train:
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        mean = X.mean(dim=(1, 2, 3), keepdim=True)
        var = ((X - mean) ** 2).mean(dim=(1, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class MyLayerNorm(Module):
    """
    Args:
         num_filters: Number of channels
    """
    def __init__(self, num_filters, batch_size=128):
        super().__init__()
        shape = (batch_size, 1, 1, 1)
        self.gamma = Parameter(torch.ones(shape))
        self.beta = Parameter(torch.zeros(shape))

        self.register_buffer('moving_mean', torch.zeros(shape), persistent=True)
        self.register_buffer('moving_var', torch.ones(shape), persistent=True)

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = layer_norm(
            x, self.gamma, self.beta, self.moving_mean, self.moving_var, train=self.training)
        return y


def group_norm(X, gamma, beta, moving_mean, moving_var, num_groups=1, eps=1e-5, momentum=0.9, train=True):
    """
    Args:
        X: Input Tensor
        gamma: Learned parameter
        beta: Learned parameter
        moving_mean: Used in test mode
        moving_var: Used in test mode.
        eps: Calculation Stabilizing Constant
        momentum: It will decide how we update moving_mean and moving_var
        train: Indicates whether training or testing
    """
    if not train:
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        batch_size = X.shape[0]
        num_channels = X.shape[1]
        num_channels_prime = num_channels // num_groups
        width = X.shape[2]
        height = X.shape[3]
        X_prime = X.reshape((batch_size, num_groups, num_channels_prime, width, height))
        mean = X_prime.mean(dim=(2, 3, 4), keepdim=True)
        var = ((X_prime - mean) ** 2).mean(dim=(2, 3, 4), keepdim=True)
        X_hat = (X_prime - mean) / torch.sqrt(var + eps)
        X_old = X_hat.reshape((batch_size, num_groups * num_channels_prime, width, height))

    else:
        batch_size = X.shape[0]
        num_channels = X.shape[1]
        num_channels_prime = num_channels // num_groups
        width = X.shape[2]
        height = X.shape[3]

        # print("X",X.shape);

        # changing to group view
        X_prime = X.reshape((batch_size, num_groups, num_channels_prime, width, height))
        mean = X_prime.mean(dim=(2, 3, 4), keepdim=True)
        var = ((X_prime - mean) ** 2).mean(dim=(2, 3, 4), keepdim=True)

        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X_prime - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean[:batch_size, :, :, :] = momentum * moving_mean[:batch_size, :, :, :] + (1.0 - momentum) * mean
        moving_var[:batch_size, :, :, :] = momentum * moving_var[:batch_size, :, :, :] + (1.0 - momentum) * var
        X_old = X_hat.reshape((batch_size, num_groups * num_channels_prime, width, height))
    Y = gamma[:batch_size, :, :, :] * X_old + beta[:batch_size, :, :, :]
    return Y, moving_mean.data, moving_var.data


class MyGroupNorm(Module):
    """
    Args:
         num_groups: Number of groups
    """

    def __init__(self, num_groups=16, batch_size=128):
        super().__init__()

        shape = (batch_size, 1, 1, 1)
        shape_prime = (batch_size, num_groups, 1, 1, 1)

        self.gamma = Parameter(torch.ones(shape))
        self.beta = Parameter(torch.zeros(shape))


        self.num_groups = num_groups


        self.register_buffer('moving_mean', torch.zeros(shape_prime), persistent=True)
        self.register_buffer('moving_var', torch.ones(shape_prime), persistent=True)

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = group_norm(
            x, self.gamma, self.beta, self.moving_mean, self.moving_var, self.num_groups, train=self.training)
        return y


class BatchInstance(Module):
    def __init__(self, num_filters, batch_size=128):
        super().__init__()
        shape = (1, num_filters, 1, 1)

        self.bn_layer = MyBatchNorm(num_filters)
        self.in_layer = MyInstanceNorm(num_filters)

        self.gamma = Parameter(torch.ones(shape))
        self.beta = Parameter(torch.zeros(shape))
        self.gate = Parameter(torch.ones(shape))
        setattr(self.gate, 'bin_gate', True)

    def forward(self, x):
        batch_normalized = self.bn_layer(x)
        instance_normalized = self.in_layer(x)
        x = (self.gate * batch_normalized + (1 - self.gate) * instance_normalized)*self.gamma + self.beta
        return x