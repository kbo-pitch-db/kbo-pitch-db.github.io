"""Equivalent Adam optimization with optional CUDA graph launch acceleration."""
import math


def optimize(parameters,optimizer,evaluate,steps,device,use_cuda_graph=False):
    import torch
    best=None;bestloss=float('inf')
    def remember(value,snapshot):
        nonlocal best,bestloss
        number=float(value.detach())
        if not math.isfinite(number):raise ValueError('Non-finite pitching objective')
        if number<bestloss:
            bestloss=number;best=[p.detach().clone() for p in snapshot]
    if not use_cuda_graph:
        for iteration in range(steps):
            optimizer.zero_grad(set_to_none=True);total,_,_=evaluate()
            remember(total,parameters);total.backward()
            torch.nn.utils.clip_grad_norm_(parameters,20.);optimizer.step()
            if iteration==int(steps*.75):
                for group in optimizer.param_groups:group['lr']*=.3
        return bestloss,best
    # Warm kernels and optimizer state on a side stream. These are the first three
    # optimization steps, not additional uncounted updates.
    with torch.cuda.device(device):
        warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False);total,_,_=evaluate();total.backward()
                torch.nn.utils.clip_grad_norm_(parameters,20.)
                remember(total,parameters);optimizer.step()
        torch.cuda.current_stream().wait_stream(warm)
        recapture=True;graphs=[]
        for iteration in range(3,steps):
            if recapture:
                graph=torch.cuda.CUDAGraph();graphs.append(graph)
                with torch.cuda.graph(graph):
                    optimizer.zero_grad(set_to_none=False);total,_,_=evaluate();total.backward()
                    torch.nn.utils.clip_grad_norm_(parameters,20.)
                    # Loss describes pre-update parameters. Preserve that exact state.
                    snapshot=[p.detach().clone() for p in parameters]
                    optimizer.step()
                recapture=False
            # Capture records work; it does not execute the optimization step.
            graph.replay()
            remember(total,snapshot)
            if iteration==int(steps*.75):
                for group in optimizer.param_groups:group['lr']*=.3
                recapture=True  # Recapture the next step at the new learning rate.
    return bestloss,best
