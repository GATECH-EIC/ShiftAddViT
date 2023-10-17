r"""
Naive gate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

r"""
Base gate with standard interface
"""
import torch.nn as nn
from torch.distributions.normal import Normal


class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None


class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, _print=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.observe = False
        self.d_model = d_model
        self._print = _print
        self.shift_ratio = None

    def forward(self, inp, ratio=None):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        # ratio = None
        gate = self.gate(inp)
        # gate = gate.reshape(-1, self.tot_expert)
        gate_top_k_val, gate_top_k_idx = torch.sort(gate, dim=-1, descending=True)
        top_k_logits = gate_top_k_val[:, : self.top_k]
        top_k_indices = gate_top_k_idx[:, : self.top_k]
        #get shift ratio
        self.shift_ratio = torch.sum(gate_top_k_idx != 0).item()/gate_top_k_idx.shape[0]
        
        # # fix ratio
        # if ratio and shift_ratio > ratio:
        #     new_gate_top_k_val = gate_top_k_val
        #     new_gate_top_k_val = torch.where(gate_top_k_idx==0, torch.tensor(-10000, dtype=new_gate_top_k_val.dtype, device=new_gate_top_k_val.device), new_gate_top_k_val)


        #     _ , shift_top_k_idx = torch.sort(new_gate_top_k_val, dim=0, descending=False)
        #     shift_top_k_idx = shift_top_k_idx[:int((1-ratio)*len(new_gate_top_k_val)),:]

        #     index = torch.squeeze(shift_top_k_idx)
        #     gate[:,1].scatter_(0, index, -10000*torch.ones_like(gate[:,1]))
        #     gate_top_k_val, gate_top_k_idx = torch.sort(gate, dim=-1, descending=True)
        #     top_k_logits = gate_top_k_val[:, : self.top_k]
        #     top_k_indices = gate_top_k_idx[:, : self.top_k]

        # elif ratio and shift_ratio < ratio:
        #     new_gate_top_k_val = gate_top_k_val

        #     new_gate_top_k_val = torch.where(gate_top_k_idx==1, torch.tensor(-10000, dtype=new_gate_top_k_val.dtype, device=new_gate_top_k_val.device), new_gate_top_k_val)
        #     _, mul_top_k_idx = torch.sort(new_gate_top_k_val, dim=0, descending=False)
        #     mul_top_k_idx = mul_top_k_idx[:int(ratio*len(new_gate_top_k_val)),:]
        #     # mul_top_k_val, mul_top_k_idx = torch.topk(
        #     #                 new_gate_top_k_val, k=int(ratio*len(new_gate_top_k_val)), dim=0, largest=False, sorted=False)
        #     index = torch.squeeze(mul_top_k_idx)
        #     gate[:,0].scatter_(0, index, -10000*torch.ones_like(gate[:,0]))
        #     gate_top_k_val, gate_top_k_idx = torch.sort(gate, dim=-1, descending=True)
        #     top_k_logits = gate_top_k_val[:, : self.top_k]
        #     top_k_indices = gate_top_k_idx[:, : self.top_k]

        # else:
        # top_k_logits = gate_top_k_val[:, : self.top_k]
        # top_k_indices = gate_top_k_idx[:, : self.top_k]
        
        if self.top_k == 1:
            top_k_gates = torch.ones_like(top_k_indices, device=top_k_indices.device,dtype=torch.float32)
        else:
            top_k_gates = F.softmax(top_k_logits, dim=-1)
        zeros = torch.zeros_like(gate, requires_grad=True).type(torch.float32)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        # if self.observe:
        #     return gates, shift_ratio

        if self._print is True:
            import numpy as np
            np.save("./visualization/gate.npy", top_k_indices.cpu())

            exit()

        return gates


class NoisyGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, _print=False):
        super().__init__(num_expert, world_size)
        self.w_gate = nn.Parameter(
            torch.zeros(d_model, self.tot_expert), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(d_model, self.tot_expert), requires_grad=True
        )
        self.top_k = top_k
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.noise_epsilon = 1e-2
        self.shift_ratio = 0

        self.reset_parameters()

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88

        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_noise, a=math.sqrt(5))


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, inp):
        clean_logits = inp @ self.w_gate
        # if self.training:
        #     raw_noise_stddev = inp @ self.w_noise
        #     noise_stddev = (
        #         self.softplus(raw_noise_stddev) + self.noise_epsilon
        #     )
        #     noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        #     logits = noisy_logits
        # else:
        #     logits = clean_logits
        raw_noise_stddev = inp @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        )
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
        
        # calculate topk + 1 that will be needed for the noisy gates
        # logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        # top_k_gates = top_k_logits

        zeros = torch.zeros_like(logits, requires_grad=True).type(torch.float32)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        # if not self.training:
        self.shift_ratio = torch.sum(top_k_indices != 0).item()/top_k_indices.shape[0]
        
        if self.top_k < self.tot_expert:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
            load[0] = load[0]*6
            load[1] = load[1]*4
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        importance[0] = importance[0]*6
        importance[1] = importance[1]*4
        loss = self.cv_squared(importance) + self.cv_squared(load)
        self.set_loss(loss)
        return gates


class NoisyGate_V2(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, _print=False):
        super().__init__(num_expert, world_size)
        self.w_gate = nn.Parameter(
            torch.zeros(d_model, self.tot_expert), requires_grad=True
        )
        # self.w_noise = nn.Parameter(
        #     torch.zeros(d_model, self.tot_expert), requires_grad=True
        # )
        self.top_k = top_k
        # self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.num_expert = num_expert
        self.noise_epsilon = 1e-2
        self.shift_ratio = 0

        self.reset_parameters()

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88

        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        # torch.nn.init.kaiming_uniform_(self.w_noise, a=math.sqrt(5))


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)
    
    def get_importance(self, logits):
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        # top_k_gates = self.softmax(top_k_logits)
        top_k_gates = top_k_logits

        zeros = torch.zeros_like(logits, requires_grad=True).type(torch.float32)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        importance = gates.sum(0)
        importance[0] = importance[0]*6
        importance[1] = importance[1]*4
        
        return importance
    
    def forward(self, inp):
        clean_logits = inp @ self.w_gate
        if self.training:
            # raw_noise_stddev = inp @ self.w_noise
            # noise_stddev = (
            #     self.softplus(raw_noise_stddev) + self.noise_epsilon
            # )
            std = 1/self.num_expert
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * std)
            logits = noisy_logits
        else:
            logits = clean_logits
        
        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        # top_k_gates = self.softmax(top_k_logits)
        top_k_gates = top_k_logits

        zeros = torch.zeros_like(logits, requires_grad=True).type(torch.float32)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        # if not self.training:
        self.shift_ratio = torch.sum(top_k_indices != 0).item()/top_k_indices.shape[0]
        
        if self.top_k < self.tot_expert and self.training:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, std, top_logits
                )
            ).sum(0)
            load[0] = load[0]*6
            load[1] = load[1]*4
        else:
            load = self._gates_to_load(gates)

        # importance = self.get_importance(clean_logits)
        importance = self.softmax(clean_logits).sum(0)
        importance[0] = importance[0]*6
        importance[1] = importance[1]*4
        
        loss = self.cv_squared(importance) + self.cv_squared(load)
        self.set_loss(loss)
        return gates


class SparseDispatcher():
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    # def __init__(self, num_experts, gates, top_k=2):
    #     """Create a SparseDispatcher."""

    #     self._gates = gates
    #     self._num_experts = num_experts

    #     if num_experts == 2 and top_k == 1: 
    #         ''' Codes here are able to have same result as dynamic computeGraph only if num_experts == 2 and top_k == 1'''
    #         ''' Use code below, we can get static computeGraph in TVM '''
    #         index = torch.arange(gates.shape[0],dtype=torch.long,device=gates.device)
    #         sorted_experts, index_sorted_experts = gates[:,1].type(torch.long).sort()

    #         expert_index = sorted_experts.unsqueeze(1)
    #         self._batch_index = index[index_sorted_experts]

    #         sorted_experts = torch.stack((index, sorted_experts), dim=1)
    #         index_sorted_experts = torch.stack((index, index_sorted_experts), dim=1)

    #     else:
    #         ''' A more versatile computing scheme '''
    #         indices = torch.where(gates != 0)
    #         nonzero_indices = torch.stack(indices, dim=1)
    #         sorted_experts, index_sorted_experts = nonzero_indices.sort(0)
    #         # get according batch index for each expert
    #         self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
    #         # drop indices
    #         _, expert_index = sorted_experts.split(1, dim=1) # get second column

    #     # calculate num samples that each expert gets
    #     self._part_sizes = (self._gates > 0).sum(0).tolist()
    #     # expand gates to match with self._batch_index
    #     gates_exp = self._gates[self._batch_index]
    #     self._nonzero_gates = torch.gather(gates_exp, 1, expert_index)


    def __init__(self, num_experts, gates, top_k=2):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        ''' A more versatile computing scheme '''
        indices = torch.where(gates != 0)
        nonzero_indices = torch.stack(indices, dim=1)
        sorted_experts, index_sorted_experts = nonzero_indices.sort(0)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # drop indices
        _, expert_index = sorted_experts.split(1, dim=1) # get second column
        # print(gates[:10,:])
        # print(self._batch_index[:10])
        # calculate num samples that each expert gets
        self._part_sizes = (self._gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = self._gates[self._batch_index]
        self._nonzero_gates = torch.gather(gates_exp, 1, expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        # print(inp[self._batch_index].shape)
        inp_exp = inp[self._batch_index].flatten(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp().type(torch.float32)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts

        index = self._batch_index.unsqueeze(-1).expand_as(stitched)
        # combined = zeros.scatter_add(0, self._batch_index.unsqueeze(-1).expand_as(stitched), stitched.float())
        combined = zeros.scatter(0, index, stitched)
        
        # add eps to all zero values in order to avoid nans when going back to log space

        combined += np.finfo(float).eps
        # back to log space
        return combined.log()
