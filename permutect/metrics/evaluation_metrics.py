import math

import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import utils
from permutect.data.read_set import ReadSetBatch
from permutect.utils import Variation


def round_up_to_nearest_three(x: int):
    return math.ceil(x / 3) * 3


def multiple_of_three_bin_index(x: int):
    return (round_up_to_nearest_three(x)//3) - 1    # -1 because zero is not a bin


def multiple_of_three_bin_index_to_count(idx: int):
    return 3 * (idx + 1)


MAX_COUNT = 18  # counts above this will be truncated
MAX_LOGIT = 6
NUM_COUNT_BINS = round_up_to_nearest_three(MAX_COUNT) // 3    # zero is not a bin

# keep track of losses during training of artifact model
class LossMetrics:
    def __init__(self, device):
        self.device = device

        self.labeled_loss = utils.StreamingAverage(device=self._device)
        self.unlabeled_loss = utils.StreamingAverage(device=self._device)

        self.labeled_loss_by_type = {variant_type: utils.StreamingAverage(device=self._device) for variant_type in Variation}
        self.labeled_loss_by_count = {bin_idx: utils.StreamingAverage(device=self._device) for bin_idx in range(NUM_COUNT_BINS)}

    def get_labeled_loss(self):
        return self.labeled_loss.get()

    def get_unlabeled_loss(self):
        return self.unlabeled_loss.get()

    def write_to_summary_writer(self, epoch_type: utils.Epoch, epoch: int, summary_writer: SummaryWriter):
        summary_writer.add_scalar(epoch_type.name + "/Labeled Loss", self.labeled_loss.get(), epoch)
        summary_writer.add_scalar(epoch_type.name + "/Unlabeled Loss", self.unlabeled_loss.get(), epoch)

        for bin_idx, loss in self.labeled_loss_by_count.items():
            summary_writer.add_scalar(
                epoch_type.name + "/Labeled Loss/By Count/" + str(multiple_of_three_bin_index_to_count(bin_idx)), loss.get(), epoch)

        for var_type, loss in self.labeled_loss_by_type.items():
            summary_writer.add_scalar(epoch_type.name + "/Labeled Loss/By Type/" + var_type.name, loss.get(), epoch)

    def record_total_batch_loss(self, total_loss: float, batch: ReadSetBatch):
        if batch.is_labeled():
            self.labeled_loss.record_sum(total_loss, batch.size())

            if batch.alt_count <= MAX_COUNT:
                self.labeled_loss_by_count[multiple_of_three_bin_index(batch.alt_count)].record_sum(total_loss, batch.size())
        else:
            self.unlabeled_loss.record_sum(total_loss, batch.size())


    def record_separate_losses(self, losses: torch.Tensor, batch: ReadSetBatch):
        if batch.is_labeled():
            types_one_hot = batch.variant_type_one_hot()
            losses_masked_by_type = losses.reshape(batch.size(), 1) * types_one_hot
            counts_by_type = torch.sum(types_one_hot, dim=0)
            total_loss_by_type = torch.sum(losses_masked_by_type, dim=0)
            variant_types = list(Variation)
            for variant_type_idx in range(len(Variation)):
                count_for_type = int(counts_by_type[variant_type_idx].item())
                loss_for_type = total_loss_by_type[variant_type_idx].item()
                self.labeled_loss_by_type[variant_types[variant_type_idx]].record_sum(loss_for_type, count_for_type)


class EvaluationMetrics:
