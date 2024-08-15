import torch
import torch.nn.functional as F
from scipy.optimize import fmin_ncg, fmin_l_bfgs_b, fmin_cg
import numpy as np
from functools import reduce

def to_vector(v):
    if isinstance(v, tuple) or isinstance(v, list):
        # return v.cpu().numpy().reshape(-1)
        return np.concatenate([vv.cpu().numpy().reshape(-1) for vv in v])
    else:
        return v.cpu().numpy().reshape(-1)

def to_list(v, sizes, device):
    _v = v
    result = []
    for size in sizes:
        total = reduce(lambda a, b: a * b, size)
        result.append(_v[:total].reshape(size).float().to(device))
        _v = _v[total:]
    return tuple(result)


def inverse_hvp_cg_sgc(data, model, edge_index, vs, lam, device):
    w = [p for p in model.parameters() if p.requires_grad][0]
    x_train = torch.tensor(data.train_set.nodes, device=device)
    y_train = torch.tensor(data.train_set.labels, device=device)
    inverse_hvp = []
    status = []
    cg_grad = []
    sizes = [p.size() for p in model.parameters() if p.requires_grad]
    v = torch.cat([vv.view(-1) for vv in vs])
    i = None
    fmin_loss_fn = _get_fmin_loss_fn_sgc(v, model=model, w=w, lam=lam,
                                         nodes=x_train, labels=y_train,
                                         edge_index=edge_index, device=device)
    fmin_grad_fn = _get_fmin_grad_fn_sgc(v, model=model, w=w, lam=lam,
                                         nodes=x_train, labels=y_train,
                                         edge_index=edge_index, device=device)
    fmin_hvp_fn = _get_fmin_hvp_fn_sgc(v, model=model, w=w, lam=lam,
                                           nodes=x_train, labels=y_train,
                                           edge_index=edge_index, device=device)
    
    res = fmin_ncg(
        f=fmin_loss_fn,
        x0=to_vector(vs),
        fprime=fmin_grad_fn,
        fhess_p=fmin_hvp_fn,
        # fhess=fmin_fhess_fn,
        # callback=cg_callback,
        avextol=1e-5,
        disp=False,
        full_output=True,
        maxiter=100)
    # inverse_hvp.append(to_list(res[0], sizes, device)[0])
    inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)
    # print('-----------------------------------')
    status = res[5]
    cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))
    return inverse_hvp, (cg_grad, status)

    


def inverse_hvp_cg(data, model, edge_index, vs, damping, device, use_torch=True):
    x_train = torch.tensor(data.train_set.nodes, device=device)
    y_train = torch.tensor(data.train_set.labels, device=device)
    inverse_hvp = []
    status = []
    cg_grad = []
    # for i, (v, p) in enumerate(zip(vs, model.parameters())):
    sizes = [p.size() for p in model.parameters() if p.requires_grad]
    # v = to_vector(vs)
    v = torch.cat([vv.view(-1) for vv in vs])
    i = None
    fmin_loss_fn = _get_fmin_loss_fn(v, model=model,
                                     x_train=x_train, y_train=y_train,
                                     edge_index=edge_index, damping=damping,
                                     sizes=sizes, p_idx=i, device=device,
                                     use_torch=use_torch)
    fmin_grad_fn = _get_fmin_grad_fn(v, model=model,
                                     x_train=x_train, y_train=y_train,
                                     edge_index=edge_index, damping=damping,
                                     sizes=sizes, p_idx=i, device=device,
                                     use_torch=use_torch)
    fmin_hvp_fn = _get_fmin_hvp_fn(v, model=model,
                                   x_train=x_train, y_train=y_train,
                                   edge_index=edge_index, damping=damping,
                                   sizes=sizes, p_idx=i, device=device,
                                   use_torch=use_torch)
    cg_callback = _get_cg_callback(v, model=model,
                                   x_train=x_train, y_train=y_train,
                                   edge_index=edge_index, damping=damping,
                                   sizes=sizes, p_idx=i, device=device,
                                   use_torch=use_torch)

    # res = minimize(fmin_loss_fn, v.view(-1), method='cg', max_iter=100)
    if use_torch:
        res = fmin_cg(
            f=fmin_loss_fn,
            x0=to_vector(vs),
            fprime=fmin_grad_fn,
            gtol=1E-4,
            # norm='fro',
            # callback=cg_callback,
            disp=False,
            full_output=True,
            maxiter=100,
        )
        # inverse_hvp.append(to_list(res[0], sizes, device)[0])
        inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)
        cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))
        status = res[4]
        # print('-----------------------------------')
        # cg_grad.append(np.linalg.norm(fmin_grad_fn(res[0]), ord=np.inf))

    else:
        res = fmin_ncg(
            f=fmin_loss_fn,
            x0=to_vector(vs),
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp_fn,
            # callback=cg_callback,
            avextol=1e-5,
            disp=False,
            full_output=True,
            maxiter=100)
        # inverse_hvp.append(to_list(res[0], sizes, device)[0])
        inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)
        # print('-----------------------------------')
        status = res[5]
        cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))

    #     x, _err, d = fmin_l_bfgs_b(
    #         func=fmin_loss_fn,
    #         x0=to_vector(v),
    #         fprime=fmin_grad_fn,
    #         iprint=0,
    #     )
    #     inverse_hvp.append(to_list(x, sizes, device)[0])
    #     status.append(d['warnflag'])
    #     err += _err.item()
    # print('error:', err, status)
    return inverse_hvp, (cg_grad, status)

def _get_fmin_loss_fn_sgc(v, **kwargs):
    model = kwargs['model']
    edge_index = kwargs['edge_index']
    w = kwargs['w']
    lam = kwargs['lam']
    nodes = kwargs['nodes']
    labels = kwargs['labels']
    device = kwargs['device']
    y = F.one_hot(labels)

    H = _hessian_sgc(model, edge_index, w, nodes, y, lam, device)
    # print('H:', H.size())

    def get_fmin_loss(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        # print(x.size)
        hvp = H.view(w.size(0), w.size(1), -1).bmm(x.view(-1, w.size(0)).t().unsqueeze(2)).squeeze().t().flatten()
        obj = 0.5 * torch.dot(hvp, x) - torch.dot(v, x)
        return obj.detach().cpu().numpy()

    return get_fmin_loss

def _get_fmin_grad_fn_sgc(v, **kwargs):
    model = kwargs['model']
    edge_index = kwargs['edge_index']
    w = kwargs['w']
    nodes = kwargs['nodes']
    labels = kwargs['labels']
    device = kwargs['device']
    lam = kwargs['lam']
    y = F.one_hot(labels)

    H = _hessian_sgc(model, edge_index, w, nodes, y, lam, device)
    def get_fmin_grad(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        # hvp = _mini_batch_hvp(x, **kwargs)
        # hvp = H.mv(x).view(-1, w.size(0)).t()
        # print(H.view(w.size(0), w.size(1), -1).bmm(x.view(-1, w.size(0)).t().unsqueeze(2)).size())
        hvp = H.view(w.size(0), w.size(1), -1).bmm(x.view(-1, w.size(0)).t().unsqueeze(2)).squeeze().t().flatten()
        # return to_vector(hvp - v.view(-1))
        return (hvp - v).cpu().numpy()

    return get_fmin_grad

def _get_fmin_hvp_fn_sgc(v, **kwargs):
    model = kwargs['model']
    edge_index = kwargs['edge_index']
    w = kwargs['w']
    nodes = kwargs['nodes']
    labels = kwargs['labels']
    device = kwargs['device']
    lam = kwargs['lam']
    y = F.one_hot(labels)

    H = _hessian_sgc(model, edge_index, w, nodes, y, lam, device)
    def get_fmin_hvp(x, p):
        p = torch.tensor(p, dtype=torch.float, device=device)
        # hvp = _mini_batch_hvp(p, **kwargs)
        with torch.no_grad():
            hvp = H.view(w.size(0), w.size(1), -1).bmm(p.view(-1, w.size(0)).t().unsqueeze(2)).squeeze().t().flatten()
        return hvp.cpu().numpy()
    return get_fmin_hvp

def _get_fmin_loss_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_loss(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        obj = 0.5 * torch.dot(torch.cat(hvp, dim=0), x) - torch.dot(v, x)
        return obj.detach().cpu().numpy()

    return get_fmin_loss

def _mini_batch_hvp(x, **kwargs):
    model = kwargs['model']
    x_train = kwargs['x_train']
    y_train = kwargs['y_train']
    edge_index = kwargs['edge_index']
    damping = kwargs['damping']
    device = kwargs['device']
    sizes = kwargs['sizes']
    p_idx = kwargs['p_idx']
    use_torch = kwargs['use_torch']

    x = to_list(x, sizes, device)
    if use_torch:
        _hvp = hessian_vector_product(model, edge_index, x_train, y_train, x, device, p_idx)
    else:
        model.eval()
        y_hat = model(x_train, edge_index)
        loss = model.loss(y_hat, y_train)
        params = [p for p in model.parameters() if p.requires_grad]
        if p_idx is not None:
            params = params[p_idx:p_idx + 1]
        _hvp = hvp(loss, params, x)
    # return _hvp[0].view(-1) + damping * x
    return [(a + damping * b).view(-1) for a, b in zip(_hvp, x)]

def hessian_vector_product(model, edge_index, x, y, v, device, p_idx=None):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if p_idx is not None:
        parameters = parameters[p_idx:p_idx+1]

    y_hat = model(x, edge_index)
    # train_loss = model.loss(y_hat, y)
    train_loss = model.loss_sum(y_hat, y)

    _, train_loss = _as_tuple(train_loss, "outputs of the user-provided function", "hvp")

    with torch.enable_grad():
        jac = _autograd_grad(train_loss, parameters, create_graph=True)
        grad_jac = tuple(
            torch.zeros_like(p, requires_grad=True, device=device) for p in parameters
        )
        double_back = _autograd_grad(jac, parameters, grad_jac, create_graph=True)

    grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=False)
    hvp = _fill_in_zeros(grad_res, parameters, False, False, "double_back_trick")
    hvp = _grad_postprocess(hvp, False)
    return hvp, train_loss[0]

def _autograd_grad(outputs, inputs, grad_outputs=None, create_graph=False, retain_graph=None):
    # Version of autograd.grad that accepts `None` in outputs and do not compute gradients for them.
    # This has the extra constraint that inputs has to be a tuple
    assert isinstance(outputs, tuple)
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    assert isinstance(grad_outputs, tuple)
    assert len(outputs) == len(grad_outputs)

    new_outputs: Tuple[torch.Tensor, ...] = tuple()
    new_grad_outputs: Tuple[torch.Tensor, ...] = tuple()
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)
            new_grad_outputs += (grad_out,)

    if len(new_outputs) == 0:
        # No differentiable output, we don't need to call the autograd engine
        return (None,) * len(inputs)
    else:
        return torch.autograd.grad(new_outputs, inputs, new_grad_outputs, allow_unused=True,
                                   create_graph=create_graph, retain_graph=retain_graph)
