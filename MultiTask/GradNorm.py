import torch

class GradNorm():
    def __init__(self, model, optimizer):
        super(GradNorm, self).__init()
        self.model = model
        self.optimizer = optimizer

    def GradNorm_train(self, i, loss):
        weighted_task_loss = torch.mul(self.model.weights, loss) # loss is the return value of the forward pass of the MultiTaskLoss
        if i = 0:
            initial_task_loss = loss.data.cpu().numpy()
        total_loss = torch.sum(weighted_task_loss)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward(retain_graph=True)
        self.model.weights.grad.data = model.weights.grad.data * 0.0

        #Pick the weights of the final layer shared between tasks
        last_layer = self.model.get_last_shared_layer()
        W = last_layer.parameters()
        #get the gradient norms for each of the tasks
        #G^{(i)}_w(t)
        norms = []
        for i in range(len(loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(loss[i], W, retain_graph=True)
            # compute the norm
            norms.append(torch.norm(torch.mul(self.model.weights[i], gygw[0])))
        norms = torch.stack(norms)

        #compute the inverse training rate r_i(t)
        loss_ratio = loss.data.cpu().numpy() / initial_task
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        #compute the mean norm
        mean_norm = np.mean(norms.data.cpu.numpy()))

        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 0.06), requires_grad=False).cuda() #0.06 is the alpha parameter

        # this is the GradNorm loss itself
        Gradloss = nn.L1Loss(reduction='sum')
        grad_norm_loss = 0
        for loss_index in range(len(loss)):
            grad_norm_loss = torch.add(grad_norm_loss, Gradloss(norms[loss_index], constant_term[loss_index]))
        # compute the gradient for the weights
        self.model.weights.grad = torch.autograd.grad(grad_norm_loss, self.model.weights)[0]

        # do a step with the optimizer
        self.optimizer.step()
