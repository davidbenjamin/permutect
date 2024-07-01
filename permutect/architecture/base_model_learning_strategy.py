from abc import ABC, abstractmethod
from typing import List

import torch.nn

from permutect.architecture.base_model import BaseModel
from permutect.architecture.mlp import MLP
from permutect.data.base_datum import BaseBatch
from permutect.parameters import BaseModelParameters


