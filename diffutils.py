import torch
import functools
import functorch


def parameter_jacobian(model, *inputs, vectorize=True, flatten=True):

    func_model, params = functorch.make_functional(model)
    #func_model_input = lambda pars: func_model(pars, *inputs)

    def func_model_input(*pars):
        """ As a function of the individual parameters"""
        y = func_model(pars, *inputs)
        y = y.reshape(-1)
        return y

    jacs = torch.autograd.functional.jacobian(func_model_input, params, vectorize=vectorize)
    n_out = jacs[0].shape[0]

    if flatten:
        jacs_2d = [jac.reshape(n_out, -1) for jac in jacs]
        J = torch.cat(jacs_2d, dim=-1)
    else:
        J = jacs

    return J