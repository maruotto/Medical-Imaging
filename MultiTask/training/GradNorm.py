import torch
import torch.nn as nn
from training import config
import torch

class GradNorm():
    __slots__ = "model", "optimizer", "initial_task_loss"
    def __init__(self, model, optimizer):
        super(GradNorm, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.initial_task_loss = 0

    def GradNorm_train(self, i, loss):
        weighted_task_loss = torch.mul(self.model.weights, loss) # loss is the return value of the forward pass of the MultiTaskLoss
        weighted_task_loss = torch.Tensor(weighted_task_loss).to(config.DEVICE)
        if i == 0:
            self.initial_task_loss = loss.data.cpu().numpy()
            self.initial_task_loss = torch.Tensor(self.initial_task_loss).to(config.DEVICE)
        total_loss = torch.sum(weighted_task_loss)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward(retain_graph=True)
        self.model.weights.grad.data = self.model.weights.grad.data * 0.0

        #Pick the weights of the final layer shared between tasks
        W = self.model.get_last_shared_layer()
        #get the gradient norms for each of the tasks
        #G^{(i)}_w(t)
        norms = []
        for i in range(len(loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(torch.mul(self.model.weights[i], gygw[0])))
        norms = torch.stack(norms)

        #compute the inverse training rate r_i(t)
        loss_ratio = torch.Tensor(loss.data.cpu().numpy()).to(config.DEVICE) / self.initial_task_loss
        inverse_train_rate = torch.Tensor(loss_ratio).to(config.DEVICE) / torch.mean(torch.Tensor(loss_ratio))

        #compute the mean norm
        mean_norm = torch.mean(torch.Tensor(norms.data.cpu().numpy()).to(config.DEVICE))

        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = (mean_norm * (inverse_train_rate ** config.ALPHA)).clone().detach().requires_grad_(False) 
        constant_term = torch.Tensor(constant_term).to(config.DEVICE)
        # this is the GradNorm loss itself
        Gradloss = nn.L1Loss(reduction='sum')
        grad_norm_loss = 0
        for loss_index in range(len(loss)):
            grad_norm_loss = torch.add(grad_norm_loss, Gradloss(norms[loss_index], constant_term[loss_index]))
        # compute the gradient for the weights
        self.model.weights.grad = torch.autograd.grad(grad_norm_loss, self.model.weights)[0]

        # do a step with the optimizer
        self.optimizer.step()
