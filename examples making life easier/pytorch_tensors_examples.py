"""
creates examples of pytorch.
"""

import torch
import numpy as np

def tensor_examples():
    """
    creates examples of tensors.
    8 types of tensors
        3 floats (ByteTensors(16 bit), FloatTensors(32 bit) & LongTensors(64 bit))
        5 ints(8-bit signed/unsigned, 16-bit, 32-bit and 64-bit)

    Inplace operation contain _(underscore) modifies tensor itself
    Functional operation leaves the original tensor untouched
    :return: None
    """
    # initializing tensors with random values
    rand_value_tensor = torch.FloatTensor(3,2)
    print("tensor initialized with values:\n ", rand_value_tensor)
    # clearing tensor content
    rand_value_tensor.zero_()
    print("tensor with zeros:\n ", rand_value_tensor)

    # tensors with python iterables
    tensor_with_py_iterables = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
    print("tensor with iterables:\n ", tensor_with_py_iterables)

    # tensors using numpy
    temp_np_array =np.zeros((3,2))
    print("numpy array with zeros:\n ", temp_np_array)

    # converting to tensor
    temp_np_array_to_tensor = torch.tensor(temp_np_array)
    print("numpy array converted to tensor:\n ", temp_np_array_to_tensor)

def tensor_operations_example():
    """
    contains some basic operations that can be performed in tensors
    :return: None
    """
    g_tensor = torch.FloatTensor([1, 2, 3])
    print(g_tensor.sum())
    print(g_tensor.mean())
    print(g_tensor.item)

if __name__ == '__main__':
    tensor_examples()
    tensor_operations_example()