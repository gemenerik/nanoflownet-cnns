# MIT License

# Copyright (c) 2017 Clément Pinard

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


def split2list(images, split, default_split=0.9):
    np.random.seed(10)
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == "1" for x in f.readlines()]
        assert len(images) == len(split_values)
    elif split is None:
        split_values = np.random.uniform(0, 1, len(images)) < default_split
    else:
        try:
            split = float(split)
        except TypeError:
            print("Invalid Split value, it must be either a filepath or a float")
            raise
        split_values = np.random.uniform(0, 1, len(images)) < split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples
