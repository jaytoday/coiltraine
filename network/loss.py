from . import loss_functional as LF
import torch
from configs import g_conf
from torch.nn import functional as F


def BCE_KLD(params):
    recon_x = params['outputs']['predictions']
    x = params['inputs']
    mu = params['outputs']['mu']
    logvar = params['outputs']['logvar']

    # the size need to be rewrote for generalization later
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3*88*200), reduction='sum')

    print(BCE)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def l1(params):
    return branched_loss(LF.l1_loss, params)


def l2(params):
    return branched_loss(LF.l2_loss, params)


def l1_attention(params):
    return branched_loss(LF.l1_attention_loss, params)


def branched_loss(loss_function, params):

    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                variable_weights: The weights for each of the variables used

                For other losses it could contain more parameters

    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """

    controls_mask = LF.compute_branches_masks(params['controls'],
                                              params['branches'][0].shape[1])
    # Update the dictionary to add also the controls mask.
    params.update({'controls_mask': controls_mask})

    # calculate loss for each branch with specific activation
    loss_branches_vec, plotable_params = loss_function(params)

    # Apply the variable weights
    # This is applied to all branches except the last one, that is the speed branch...
    # TODO This is hardcoded to  have 4 branches not using speed.

    for i in range(4):
        loss_branches_vec[i] = loss_branches_vec[i][:, 0] * params['variable_weights']['Steer'] \
                               + loss_branches_vec[i][:, 1] * params['variable_weights']['Gas'] \
                               + loss_branches_vec[i][:, 2] * params['variable_weights']['Brake']

    loss_function = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + \
                    loss_branches_vec[3]

    speed_loss = loss_branches_vec[4]/(params['branches'][0].shape[0])

    return torch.sum(loss_function) / (params['branches'][0].shape[0])\
                + torch.sum(speed_loss) / (params['branches'][0].shape[0]),\
           plotable_params


def Loss(loss_name):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if g_conf.MODEL_TYPE == 'VAE':

        if loss_name == 'BCE+KLD':

            return BCE_KLD

        else:
            raise ValueError(" Not found Loss name")


    else:

        if loss_name == 'L1':

            return l1

        elif loss_name == 'L2':

            return l2

        else:
            raise ValueError(" Not found Loss name")


