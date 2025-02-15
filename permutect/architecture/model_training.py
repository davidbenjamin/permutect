import math
import time
from collections import defaultdict
from queue import PriorityQueue
from typing import List

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from permutect.architecture.balancer import Balancer
from permutect.architecture.permutect_model import PermutectModel, record_embeddings
from permutect.data.features_dataset import FeaturesDataset
from permutect.data.reads_batch import DownsampledReadsBatch
from permutect.data.reads_dataset import ReadsDataset, ALL_COUNTS_INDEX
from permutect.data.features_batch import FeaturesBatch
from permutect.data.datum import Datum
from permutect.data.prefetch_generator import prefetch_generator
from permutect.metrics.evaluation_metrics import EmbeddingMetrics, round_up_to_nearest_three, MAX_COUNT, \
    EvaluationMetrics
from permutect.metrics.loss_metrics import BatchIndexedAverages, BatchProperty
from permutect.parameters import TrainingParameters
from permutect.misc_utils import report_memory_usage, backpropagate, freeze, \
    unfreeze
from permutect.utils.enums import Variation, Epoch, Label

WORST_OFFENDERS_QUEUE_SIZE = 100


def train_permutect_model(model: PermutectModel, dataset: ReadsDataset, training_params: TrainingParameters,
                          summary_writer: SummaryWriter, validation_fold: int = None):
    report_memory_usage("Beginning training.")
    device, dtype = model._device, model._dtype
    num_sources = dataset.max_source + 1
    is_cuda = device.type == 'cuda'
    print(f"Is CUDA available? {is_cuda}")

    dataset.report_totals()

    alt_count_loss_func = torch.nn.MSELoss(reduction='none')

    train_optimizer = torch.optim.AdamW(model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay)
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        train_optimizer, factor=0.2, patience=5, threshold=0.001, min_lr=(training_params.learning_rate/100), verbose=True)

    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    balancer = Balancer(dataset).to(device=device, dtype=dtype)

    validation_fold_to_use = (dataset.num_folds - 1) if validation_fold is None else validation_fold
    train_loader, valid_loader = dataset.make_train_and_valid_loaders(validation_fold_to_use, training_params.batch_size, is_cuda, training_params.num_workers)

    for epoch in trange(1, training_params.num_epochs + 1, desc="Epoch"):
        model.alt_count_predictor.set_adversarial_strength((2/(1 + math.exp(-0.1*(epoch - 1)))) - 1) # alpha increases linearly
        start_epoch = time.time()
        report_memory_usage(f"Start of epoch {epoch}")
        for epoch_type in (Epoch.TRAIN, Epoch.VALID):
            model.set_epoch_type(epoch_type)
            loss_metrics = BatchIndexedAverages(num_sources=num_sources, device=device)
            alt_count_loss_metrics = BatchIndexedAverages(num_sources=num_sources, device=device)    # loss on the adversarial task

            loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader

            for original_batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
                batch = DownsampledReadsBatch(original_batch)
                # TODO: use the weight-balancing scheme that artifact model training uses
                weights = balancer.calculate_batch_weights(batch)

                representations, _ = model.calculate_representations(batch, weight_range=model._params.reweighting_range)
                calibrated_logits, uncalibrated_logits = model.logits_from_reads_batch(representations, batch)

                # TODO: code duplication with artifact model training
                # TODO: should we use calibrated logits?
                # base batch always has labels, but for unlabeled elements these labels are meaningless and is_labeled_mask is zero
                cross_entropies = bce(uncalibrated_logits, batch.get_training_labels())
                probabilities = torch.sigmoid(uncalibrated_logits)
                entropies = bce(uncalibrated_logits, probabilities)

                semisupervised_losses = batch.get_is_labeled_mask() * cross_entropies + (1 - batch.get_is_labeled_mask()) * entropies
                loss_metrics.record(batch, semisupervised_losses, weights)

                # TODO: use nonlinear transformation of counts
                # TODO: should alt count adversarial losses have label-balancing weights, too? (probably yes)
                alt_count_pred = torch.sigmoid(model.alt_count_predictor.adversarial_forward(representations).squeeze())
                alt_count_target = batch.get_alt_counts().to(dtype=alt_count_pred.dtype)/20
                alt_count_losses = alt_count_loss_func(alt_count_pred, alt_count_target)
                alt_count_loss_metrics.record(batch, values=alt_count_losses, weights=torch.ones_like(alt_count_losses))

                loss = torch.sum((weights * semisupervised_losses) + alt_count_losses)

                if epoch_type == Epoch.TRAIN:
                    backpropagate(train_optimizer, loss)

            # done with one epoch type -- training or validation -- for this epoch
            loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="semi-supervised-loss")
            alt_count_loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="alt-count-loss")

            if epoch_type == Epoch.TRAIN:
                mean_over_labels = torch.mean(loss_metrics.get_marginal(BatchProperty.LABEL)).item()
                train_scheduler.step(mean_over_labels)

            loss_metrics.report_marginals(f"Semisupervised losses for {epoch_type.name} epoch {epoch}.")
            alt_count_loss_metrics.report_marginals(f"Alt count prediction adversarial task loss for {epoch_type.name} epoch {epoch}.")
        report_memory_usage(f"End of epoch {epoch}.")
        print(f"time elapsed(s): {time.time() - start_epoch:.1f}")
        # done with training and validation for this epoch
        # note that we have not learned the AF spectrum yet
    # done with training

    record_embeddings(model, train_loader, summary_writer)


# TODO: rename this to refine on features dataset or just refine
def train_on_artifact_dataset(model: PermutectModel, dataset: FeaturesDataset, training_params: TrainingParameters, summary_writer: SummaryWriter,
                              validation_fold: int = None, epochs_per_evaluation: int = None, calibration_sources: List[int] = None):
    device, dtype = model._device, model._dtype
    bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
    balancer = Balancer(dataset, learning_rate=training_params.learning_rate).to(device=device, dtype=dtype)

    num_sources = dataset.validate_sources()
    model.reset_source_predictor(num_sources)

    train_optimizer = torch.optim.AdamW(model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay)
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(train_optimizer, factor=0.2, patience=5,
        threshold=0.001, min_lr=(training_params.learning_rate / 100), verbose=True)

    num_sources = len(dataset.totals_sclt)
    for source in range(num_sources):
        print(f"Data counts for source {source}:")
        for var_type in Variation:
            print(f"Data counts for variant type {var_type.name}:")
            for label in Label:
                print(f"{label.name}: {int(dataset.totals_sclt[source][ALL_COUNTS_INDEX][label][var_type].item())}")

    is_cuda = device.type == 'cuda'
    print(f"Is CUDA available? {is_cuda}")

    validation_fold_to_use = (dataset.num_folds - 1) if validation_fold is None else validation_fold
    train_loader = dataset.make_data_loader(dataset.all_but_one_fold(validation_fold_to_use), training_params.batch_size, is_cuda, training_params.num_workers)
    report_memory_usage(f"Train loader created.")
    valid_loader = dataset.make_data_loader([validation_fold_to_use], training_params.inference_batch_size, is_cuda, training_params.num_workers)
    report_memory_usage(f"Validation loader created.")

    calibration_train_loader = train_loader if calibration_sources is None else \
        dataset.make_data_loader(dataset.all_but_one_fold(validation_fold_to_use), training_params.batch_size,
                                 is_cuda, training_params.num_workers, sources_to_use=calibration_sources)

    calibration_valid_loader = valid_loader if calibration_sources is None else \
        dataset.make_data_loader([validation_fold_to_use], training_params.inference_batch_size,
                                 is_cuda, training_params.num_workers, sources_to_use=calibration_sources)

    first_epoch, last_epoch = 1, training_params.num_epochs + training_params.num_calibration_epochs
    for epoch in trange(1, last_epoch + 1, desc="Epoch"):
        start_of_epoch = time.time()
        report_memory_usage(f"Epoch {epoch}.")
        is_calibration_epoch = epoch > training_params.num_epochs

        model.source_predictor.set_adversarial_strength((2 / (1 + math.exp(-0.1 * (epoch - 1)))) - 1)

        for epoch_type in [Epoch.TRAIN, Epoch.VALID]:
            model.set_epoch_type(epoch_type)
            # in calibration epoch, freeze the model except for calibration
            if is_calibration_epoch and epoch_type == Epoch.TRAIN:
                freeze(model.parameters())
                #unfreeze(self.calibration_parameters())  # unfreeze calibration but everything else stays frozen
                unfreeze(model.final_calibration_shift_parameters())  # unfreeze final calibration shift but everything else stays frozen

            loss_metrics = BatchIndexedAverages(num_sources=num_sources, device=device)   # based on calibrated logits
            source_prediction_loss_metrics = BatchIndexedAverages(num_sources=num_sources, device=device)  # based on calibrated logits

            loader = (calibration_train_loader if epoch_type == Epoch.TRAIN else calibration_valid_loader) if is_calibration_epoch else \
                (train_loader if epoch_type == Epoch.TRAIN else valid_loader)

            batch: FeaturesBatch
            for batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
                sources = batch.get_sources()
                labels = batch.get_training_labels()
                logits, precalibrated_logits = model.logits_from_features_batch(batch)

                # one-hot prediction of sources
                if num_sources > 1:
                    source_prediction_logits = model.source_predictor.adversarial_forward(batch.get_representations_2d())
                    source_prediction_probs = torch.nn.functional.softmax(source_prediction_logits, dim=-1)
                    source_prediction_targets = torch.nn.functional.one_hot(sources.long(), num_sources)
                    source_prediction_losses = torch.sum(torch.square(source_prediction_probs - source_prediction_targets), dim=-1)

                    # TODO: always by count?
                    source_prediction_weights = balancer.calculate_batch_source_weights(batch, by_count=is_calibration_epoch)
                else:
                    source_prediction_losses = torch.zeros_like(logits, device=device)
                    source_prediction_weights = torch.zeros_like(logits, device=device)

                uncalibrated_cross_entropies = bce(precalibrated_logits, labels)
                calibrated_cross_entropies = bce(logits, labels)

                # TODO: investigate whether using the average of un-calibrated and calibrated cross entropies is
                # TODO: really the right thing to do
                labeled_losses = batch.get_is_labeled_mask() * (uncalibrated_cross_entropies + calibrated_cross_entropies) / 2

                # unlabeled loss: entropy regularization. We use the uncalibrated logits because otherwise entropy
                # regularization simply biases calibration to be overconfident.
                probabilities = torch.sigmoid(precalibrated_logits)
                entropies = torch.nn.functional.binary_cross_entropy_with_logits(precalibrated_logits, probabilities, reduction='none')
                unlabeled_losses = (1 - batch.get_is_labeled_mask()) * entropies

                # this updates the autobalancing as a side effect
                weights = balancer.calculate_autobalancing_weights(batch, probabilities)

                # these losses include weights and take labeled vs unlabeled into account
                losses = (labeled_losses + unlabeled_losses) * weights + (source_prediction_losses * source_prediction_weights)
                loss = torch.sum(losses)

                loss_metrics.record(batch, labeled_losses + unlabeled_losses, weights)
                source_prediction_loss_metrics.record(batch, source_prediction_losses, source_prediction_weights)

                # calibration epochs freeze the model up to calibration, so I wonder if a purely unlabeled batch
                # would cause lack of gradient problems. . .
                if epoch_type == Epoch.TRAIN:
                    backpropagate(train_optimizer, loss)

            # done with one epoch type -- training or validation -- for this epoch
            loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="semisupervised-loss")
            source_prediction_loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="source-loss")
            if epoch_type == Epoch.TRAIN:
                mean_over_labels = torch.mean(loss_metrics.get_marginal(BatchProperty.LABEL)).item()
                train_scheduler.step(mean_over_labels)

            loss_metrics.report_marginals(f"Semisupervised loss for {epoch_type.name} epoch {epoch}.")
            source_prediction_loss_metrics.report_marginals(f"Source prediction loss for {epoch_type.name} epoch {epoch}.")
        # done with training and validation for this epoch
        report_memory_usage(f"End of epoch {epoch}.")
        print(f"Time elapsed(s): {time.time() - start_of_epoch:.1f}")
        if (epochs_per_evaluation is not None and epoch % epochs_per_evaluation == 0) or (epoch == last_epoch):
            print(f"performing evaluation on epoch {epoch}")
            evaluate_model(model, epoch, dataset, train_loader, valid_loader, summary_writer, collect_embeddings=False, report_worst=False)

        # note that we have not learned the AF spectrum yet
    # done with training


@torch.inference_mode()
def collect_evaluation_data(model: PermutectModel, dataset: FeaturesDataset, train_loader, valid_loader, report_worst: bool):
    # the keys are tuples of (true label -- 1 for variant, 0 for artifact; rounded alt count)
    worst_offenders_by_truth_and_alt_count = defaultdict(lambda: PriorityQueue(WORST_OFFENDERS_QUEUE_SIZE))

    balancer = Balancer(dataset).to(device=model._device, dtype=model._dtype)
    evaluation_metrics = EvaluationMetrics()
    epoch_types = [Epoch.TRAIN, Epoch.VALID]
    for epoch_type in epoch_types:
        assert epoch_type == Epoch.TRAIN or epoch_type == Epoch.VALID  # not doing TEST here
        loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader

        batch: FeaturesBatch
        for batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
            # these are the same weights used in training
            # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
            weights = balancer.calculate_batch_weights(batch).cpu()     # not on GPU!

            logits, _ = model.logits_from_features_batch(batch)
            # logits are calculated on the GPU (when available), so we must detach AND send back to CPU (if applicable)
            pred = logits.detach().cpu()

            labels = batch.get_training_labels().cpu()
            correct = ((pred > 0) == (labels > 0.5)).tolist()

            for datum_array, predicted_logit, correct_call, weight in zip(batch.get_data_2d(),
                    pred.tolist(), correct, weights.tolist()):
                datum = Datum(datum_array)
                label = Label(datum.get_label())
                alt_count = datum.get_alt_count()
                evaluation_metrics.record_call(epoch_type, datum.get_variant_type(), predicted_logit, label, correct_call,
                                               alt_count, weight, source=datum.get_source())

                if (label != Label.UNLABELED) and report_worst and not correct_call:
                    rounded_count = round_up_to_nearest_three(alt_count)
                    label_name = Label.ARTIFACT.name if label > 0.5 else Label.VARIANT.name
                    confidence = abs(predicted_logit)

                    # the 0th aka highest priority element in the queue is the one with the lowest confidence
                    pqueue = worst_offenders_by_truth_and_alt_count[(label_name, rounded_count)]

                    # clear space if this confidence is more egregious
                    if pqueue.full() and pqueue.queue[0][0] < confidence:
                        pqueue.get()  # discards the least confident bad call

                    if not pqueue.full():  # if space was cleared or if it wasn't full already
                        pqueue.put((confidence, str(datum.get_contig()) + ":" + str(
                            datum.get_position()) + ':' + datum.get_ref_allele() + "->" + datum.get_alt_allele()))
        # done with this epoch type
    # done collecting data
    return evaluation_metrics, worst_offenders_by_truth_and_alt_count


@torch.inference_mode()
def evaluate_model(model: PermutectModel, epoch: int, dataset: FeaturesDataset, train_loader, valid_loader,
                   summary_writer: SummaryWriter, collect_embeddings: bool = False, report_worst: bool = False):

    # self.freeze_all()
    evaluation_metrics, worst_offenders_by_truth_and_alt_count = collect_evaluation_data(model, dataset, train_loader, valid_loader, report_worst)
    evaluation_metrics.make_plots(summary_writer, epoch=epoch)

    if report_worst:
        for (true_label, rounded_count), pqueue in worst_offenders_by_truth_and_alt_count.items():
            tag = "True label: " + true_label + ", rounded alt count: " + str(rounded_count)

            lines = []
            while not pqueue.empty():   # this goes from least to most egregious, FYI
                confidence, var_string = pqueue.get()
                lines.append(f"{var_string} ({confidence:.2f})")

            summary_writer.add_text(tag, "\n".join(lines), global_step=epoch)

    if collect_embeddings:
        embedding_metrics = EmbeddingMetrics()

        # now go over just the validation data and generate feature vectors / metadata for tensorboard projectors (UMAP)
        batch: FeaturesBatch
        for batch in tqdm(prefetch_generator(valid_loader), mininterval=60, total=len(valid_loader)):
            logits, _ = model.logits_from_features_batch(batch)
            pred = logits.detach().cpu()
            labels = batch.get_training_labels().cpu()
            correct = ((pred > 0) == (labels > 0.5)).tolist()
            is_labeled_list = batch.get_is_labeled_mask().cpu().tolist()

            label_strings = [("artifact" if label > 0.5 else "non-artifact") if is_labeled > 0.5 else "unlabeled"
                             for (label, is_labeled) in zip(labels.tolist(), is_labeled_list)]

            correct_strings = [str(correctness) if is_labeled > 0.5 else "-1"
                             for (correctness, is_labeled) in zip(correct, is_labeled_list)]

            for (metrics, embedding) in [(embedding_metrics, batch.get_representations_2d().detach().cpu())]:
                metrics.label_metadata.extend(label_strings)
                metrics.correct_metadata.extend(correct_strings)
                metrics.type_metadata.extend([Variation(idx).name for idx in batch.get_variant_types().cpu().tolist()])
                metrics.truncated_count_metadata.extend([str(round_up_to_nearest_three(min(MAX_COUNT, alt_count))) for alt_count in batch.get_alt_counts().cpu().tolist()])
                metrics.representations.append(embedding)
        embedding_metrics.output_to_summary_writer(summary_writer, epoch=epoch)
    # done collecting data


# TODO: these were copied from artifact model and probably could use some refactoring
def evaluate_model_after_training(model: PermutectModel, dataset: FeaturesDataset, batch_size, num_workers, summary_writer: SummaryWriter):
    train_loader = dataset.make_data_loader(dataset.all_but_the_last_fold(), batch_size, model._device.type == 'cuda', num_workers)
    valid_loader = dataset.make_data_loader(dataset.last_fold_only(), batch_size, model._device.type == 'cuda', num_workers)
    evaluate_model(model, None, dataset, train_loader, valid_loader, summary_writer, collect_embeddings=True, report_worst=True)

