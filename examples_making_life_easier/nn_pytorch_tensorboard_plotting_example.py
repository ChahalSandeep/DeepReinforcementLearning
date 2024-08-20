"""
contains some basic examples of pytorch plotting with tensorboard
"""

import math
from torch.utils.tensorboard import SummaryWriter



def writer_example():
    """
    uses SummaryWriter to write tensorboard summaries under the runs folder for every launch
    names - name of new directory include current date and time and hostname
    override name - pass log_dir argument to SummaryWriter
        can add suffix to name of directory by passing a comment option.

    contains example to capture different semantics
    periodical flush happens every 2 minutes
    :return: None
    """
    writer = SummaryWriter()
    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

    for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)
    writer.close()


if __name__ == "__main__":
    writer_example()


