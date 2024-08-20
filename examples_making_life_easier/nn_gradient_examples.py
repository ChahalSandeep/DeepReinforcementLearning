"""
contains information about gradients and tensors.
Additionally, explains static and dynamic graph.
Static Graph:define calculations in advance and cant be change later
Dynamic Graph: just do operation, when asked to calculate gradient it unrolls history of operations, accumulating
 gradient of network parameters

 Attributes related to gradient calculations:
 - grad: property which holds a tensor of the same shape computed gradient
 - is_leaf:True  if this tensor was constructed by user or if object is result of function transformation.
 - requires_grad:True if tensor required gradient calculation. inherited from is_leaf which gets this value from tensor
  constructions step. by default is False.
"""
import torch

v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])

v_sum = v1 + v2
v_res = (v_sum * 2).sum()
print("result of functions: ", v_res)
print("gradient attributes \n v1 :, %r , v2: %r, v_sum: %r, v_res %r" % (v1.is_leaf, v2.is_leaf,
                                                                         v_sum.is_leaf, v_res.is_leaf))
print("variables requiring gradient : v1: %r, v2: %r,  v_sum: %r, v_res: %r" % (v1.requires_grad, v2.requires_grad,
      v_sum.requires_grad, v_res.requires_grad))
v_res.backward()

print("gradient calculations shows v2 grad not calculated : v1: %r, v2: %r,  v_sum: %r, v_res: %r" % (
    v1.grad, v2.grad, v_sum.grad, v_res.grad))
