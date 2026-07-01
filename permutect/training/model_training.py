from __future__ import annotations

import math
import time
from collections import defaultdict
from queue import PriorityQueue
from typing import Any
from typing import List

import torch
from torch import device
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange

from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.artifact_model import record_embeddings
from permutect.data.batch import Batch
from permutect.data.batch import BatchProperty
from permutect.data.batch import DownsampledBatch
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import alt_count_bin_name
from permutect.data.count_binning import round_alt_count_to_bin_center
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.prefetch_generator import prefetch_generator
from permutect.data.reads_dataset import ReadsDataset
from permutect.metrics.evaluation_metrics import EmbeddingMetrics
from permutect.metrics.evaluation_metrics import EvaluationMetrics
from permutect.misc_utils import Timer
from permutect.misc_utils import backpropagate
from permutect.misc_utils import check_for_nan
from permutect.misc_utils import report_memory_usage
from permutect.parameters import TrainingParameters
from permutect.training.balancer import Balancer
from permutect.training.balancer import PlotType
from permutect.training.checkpoint import Checkpoint
from permutect.training.downsampler import Downsampler
from permutect.training.loss_recorder import LossRecorder
from permutect.utils.enums import Epoch
from permutect.utils.enums import Label
from permutect.utils.enums import ParameterSet
from permutect.utils.enums import Variation

WORST_OFFENDERS_QUEUE_SIZE = 100


def train_artifact_model(
    model: ArtifactModel,
    train_dataset: ReadsDataset,
    valid_dataset: ReadsDataset | None,
    training_params: TrainingParameters,
    summary_writer: SummaryWriter,
    epochs_per_evaluation: int = 5,
    trainable_params: List[ParameterSet] = None,
):
    device, dtype = model._device, model._dtype
    balancer = Balancer(num_sources=train_dataset.num_sources(), device=device).to(device=device, dtype=dtype)
    downsampler: Downsampler = Downsampler(num_sources=train_dataset.num_sources()).to(device=device, dtype=dtype)
    downsampler.optimize_downsampling_balance(train_dataset.totals_slvra.to(device=device))

    num_sources = train_dataset.validate_sources()
    train_dataset.report_totals()
    is_cuda = device.type == "cuda"
    print(f"Is CUDA available? {is_cuda}")

    train_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params.learning_rate,
        weight_decay=training_params.weight_decay,
    )
    scheduler_kwargs = {
        "factor": 0.2,
        "patience": 5,
        "threshold": 0.001,
        "min_lr": (training_params.learning_rate / 100),
    }
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(train_optimizer, **scheduler_kwargs)

    checkpoint = Checkpoint(device, model, train_optimizer)

    train_loader = train_dataset.make_data_loader(training_params.batch_size, is_cuda, training_params.num_workers)
    valid_loader = (
        None
        if valid_dataset is None
        else valid_dataset.make_data_loader(training_params.batch_size, is_cuda, training_params.num_workers)
    )
    report_memory_usage("Loaders created, about to train.")

    first_epoch, last_epoch = 1, training_params.num_epochs
    for epoch in trange(1, last_epoch + 1, desc="Epoch"):
        start_of_epoch = time.time()
        report_memory_usage(f"Epoch {epoch}.")

        for epoch_type in [Epoch.TRAIN] if valid_loader is None else [Epoch.TRAIN, Epoch.VALID]:
            train_one_epoch(
                balancer,
                checkpoint,
                device,
                downsampler,
                epoch,
                epoch_type,
                epochs_per_evaluation,
                last_epoch,
                model,
                num_sources,
                summary_writer,
                train_loader,
                train_optimizer,
                train_scheduler,
                valid_loader,
                trainable_params,
            )

        # done with training and validation for this epoch
        report_memory_usage(f"End of epoch {epoch}.")
        print(f"Time elapsed(s): {time.time() - start_of_epoch:.1f}")
        # note that we have not learned the AF spectrum yet
    # done with training
    report_memory_usage("Training complete, recording embeddings for tensorboard.")
    embeddings_timer = Timer("Creating training and validation datasets")
    record_embeddings(model, train_loader, summary_writer)
    embeddings_timer.report("Time to record embeddings for tensorboard.")


def train_one_epoch(
    balancer: Balancer,
    checkpoint: Checkpoint,
    device: device,
    downsampler: Downsampler,
    epoch: int,
    epoch_type: Epoch,
    epochs_per_evaluation: int,
    last_epoch: int,
    model: ArtifactModel,
    num_sources: int,
    summary_writer: SummaryWriter,
    train_loader: DataLoader[Any],
    train_optimizer: AdamW,
    train_scheduler: ReduceLROnPlateau,
    valid_loader: DataLoader[Any],
    trainable_params: List[ParameterSet] = None,
):
    loss_recorder = LossRecorder(device, num_sources)
    model.set_epoch_type(epoch_type, trainable_params)
    loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader

    parent_batch: Batch
    for parent_batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
        batch: DownsampledBatch
        for downsampling_iteration in range(2):
            ref_fracs_b, alt_fracs_b = downsampler.calculate_downsampling_fractions(parent_batch)
            batch = DownsampledBatch(parent_batch, ref_fracs_b, alt_fracs_b)
            output = model.compute_batch_output(batch, balancer)
            losses = model.compute_batch_losses(output, batch)
            loss_recorder.record(output, losses, batch)

            if epoch_type == Epoch.TRAIN:
                average_loss = losses.total_loss.item() / batch.size()
                if epoch > 1 and average_loss > 100.0:
                    print(f"Very large batch loss {average_loss:.2f}.")

                backpropagate(train_optimizer, losses.total_loss, params_to_clip=model.parameters())
        # done with this downsampled batch
    # done with this parent batch
    check_for_nan(model)
    if epoch_type == Epoch.TRAIN:
        mean_loss = torch.mean(loss_recorder.primary_metrics.get_marginal(BatchProperty.LABEL)).item()
        train_scheduler.step(mean_loss)

    generate_plots = epoch % epochs_per_evaluation == 0 or epoch == last_epoch
    loss_recorder.output_results(epoch_type, epoch, summary_writer, generate_plots)

    if generate_plots:
        weight_prefix = "log(label-balancing weights)" + f"({epoch_type.name})"
        count_prefix = "unweighted data counts after downsampling" + f"({epoch_type.name})"
        balancer.make_plots(summary_writer, weight_prefix, epoch, PlotType.WEIGHTS)
        balancer.make_plots(summary_writer, count_prefix, epoch, PlotType.COUNTS)

    print(f"performing evaluation on epoch {epoch}")
    if epoch_type == Epoch.VALID:
        evaluate_model(
            model,
            epoch,
            num_sources,
            balancer,
            downsampler,
            train_loader,
            valid_loader,
            summary_writer,
            collect_embeddings=False,
            report_worst=False,
        )

    if epoch_type == Epoch.TRAIN:
        mean_loss = torch.mean(loss_recorder.primary_metrics.get_marginal(BatchProperty.LABEL))
        checkpoint.save_checkpoint_if_needed(epoch, mean_loss)
        checkpoint.load_checkpoint_if_needed(mean_loss)


@torch.inference_mode()
def collect_evaluation_data(
    model: ArtifactModel,
    num_sources: int,
    balancer: Balancer,
    downsampler: Downsampler,
    train_loader,
    valid_loader,
    report_worst: bool,
):
    # the keys are tuples of (Label; rounded alt count)
    worst_offenders_by_label_and_alt_count = defaultdict(lambda: PriorityQueue(WORST_OFFENDERS_QUEUE_SIZE))

    evaluation_metrics = EvaluationMetrics(num_sources=num_sources, device=model._device)
    epoch_types = [Epoch.TRAIN, Epoch.VALID]
    for epoch_type in epoch_types:
        assert epoch_type == Epoch.TRAIN or epoch_type == Epoch.VALID  # not doing TEST here
        loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader

        parent_batch: Batch
        for parent_batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
            # TODO: magic constant
            for _ in range(3):
                ref_fracs_b, alt_fracs_b = downsampler.calculate_downsampling_fractions(parent_batch)
                batch = DownsampledBatch(parent_batch, ref_fracs_b=ref_fracs_b, alt_fracs_b=alt_fracs_b)
                output = model.compute_batch_output(batch, balancer)

                evaluation_metrics.record_batch(epoch_type, batch, logits=output.logits_b, weights=output.weights)

                if report_worst:
                    for int_array, float_array, predicted_logit in zip(
                        batch.get_int_array_be(),
                        batch.get_float_array_be(),
                        output.logits_b.detach().cpu().tolist(),
                    ):
                        datum = Datum(int_array, float_array)
                        wrong_call = (datum.get(Data.LABEL) == Label.ARTIFACT and predicted_logit < 0) or (
                            datum.get(Data.LABEL) == Label.VARIANT and predicted_logit > 0
                        )
                        if wrong_call:
                            alt_count = datum.get(Data.ALT_COUNT)
                            rounded_count = round_alt_count_to_bin_center(alt_count)
                            confidence = abs(predicted_logit)

                            # the 0th aka highest priority element in the queue is the one with the lowest confidence
                            pqueue = worst_offenders_by_label_and_alt_count[
                                (Label(datum.get(Data.LABEL)), rounded_count)
                            ]

                            # clear space if this confidence is more egregious
                            if pqueue.full() and pqueue.queue[0][0] < confidence:
                                pqueue.get()  # discards the least confident bad call

                            if not pqueue.full():  # if space was cleared or if it wasn't full already
                                pqueue.put(
                                    (
                                        confidence,
                                        str(datum.get(Data.CONTIG))
                                        + ":"
                                        + str(datum.get(Data.POSITION))
                                        + ":"
                                        + datum.get_ref_allele()
                                        + "->"
                                        + datum.get_alt_allele(),
                                    )
                                )
        # done with this epoch type
    # done collecting data
    return evaluation_metrics, worst_offenders_by_label_and_alt_count


@torch.inference_mode()
def evaluate_model(
    model: ArtifactModel,
    epoch: int,
    num_sources: int,
    balancer: Balancer,
    downsampler: Downsampler,
    train_loader,
    valid_loader,
    summary_writer: SummaryWriter,
    collect_embeddings: bool = False,
    report_worst: bool = False,
):
    # self.freeze_all()
    evaluation_metrics, worst_offenders_by_label_and_alt_count = collect_evaluation_data(
        model, num_sources, balancer, downsampler, train_loader, valid_loader, report_worst
    )
    evaluation_metrics.put_on_cpu()
    evaluation_metrics.make_plots(summary_writer, epoch=epoch)

    if report_worst:
        for (true_label, rounded_count), pqueue in worst_offenders_by_label_and_alt_count.items():
            tag = f"True label: {true_label.name}, rounded alt count: {rounded_count}"

            lines = []
            while not pqueue.empty():  # this goes from least to most egregious, FYI
                confidence, var_string = pqueue.get()
                lines.append(f"{var_string} ({confidence:.2f})")

            summary_writer.add_text(tag, "\n".join(lines), global_step=epoch)

    if collect_embeddings:
        embedding_metrics = EmbeddingMetrics()

        # now go over just the validation data and generate feature vectors / metadata for tensorboard projectors
        batch: Batch
        for batch in tqdm(prefetch_generator(valid_loader), mininterval=60, total=len(valid_loader)):
            output = model.compute_batch_output(batch)
            pred_b = output.logits_b.detach().cpu()
            labels_b = batch.get_training_labels().cpu()
            correct_b = ((pred_b > 0) == (labels_b > 0.5)).tolist()
            is_labeled_list = batch.get_is_labeled_mask().cpu().tolist()

            label_strings = [
                ("artifact" if label > 0.5 else "non-artifact") if is_labeled > 0.5 else "unlabeled"
                for (label, is_labeled) in zip(labels_b.tolist(), is_labeled_list)
            ]

            correct_strings = [
                str(correctness) if is_labeled > 0.5 else "-1"
                for (correctness, is_labeled) in zip(correct_b, is_labeled_list)
            ]

            embedding_metrics.label_metadata.extend(label_strings)
            embedding_metrics.correct_metadata.extend(correct_strings)
            embedding_metrics.type_metadata.extend(
                [Variation(idx).name for idx in batch.get(Data.VARIANT_TYPE).cpu().tolist()]
            )
            embedding_metrics.truncated_count_metadata.extend(
                [
                    alt_count_bin_name(alt_count_bin_index(alt_count))
                    for alt_count in batch.get(Data.ALT_COUNT).cpu().tolist()
                ]
            )
            embedding_metrics.features.append(output.features_be.detach().cpu())
            embedding_metrics.ref_features.append(output.ref_features_be.detach().cpu())
        embedding_metrics.output_to_summary_writer(summary_writer, epoch=epoch)
    # done collecting data
