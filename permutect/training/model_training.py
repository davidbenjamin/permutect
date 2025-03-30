import math
import random
import time
from collections import defaultdict
from queue import PriorityQueue
from typing import List

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from permutect.training.balancer import Balancer
from permutect.training.downsampler import Downsampler
from permutect.architecture.artifact_model import ArtifactModel, record_embeddings
from permutect.data.reads_batch import DownsampledReadsBatch, ReadsBatch
from permutect.data.reads_dataset import ReadsDataset
from permutect.data.datum import Datum
from permutect.data.prefetch_generator import prefetch_generator
from permutect.metrics.evaluation_metrics import EmbeddingMetrics, EvaluationMetrics
from permutect.metrics.loss_metrics import LossMetrics
from permutect.data.batch import BatchProperty
from permutect.data.count_binning import alt_count_bin_index, round_alt_count_to_bin_center, alt_count_bin_name
from permutect.parameters import TrainingParameters
from permutect.misc_utils import report_memory_usage, backpropagate, freeze, unfreeze
from permutect.utils.enums import Variation, Epoch, Label

WORST_OFFENDERS_QUEUE_SIZE = 100


def train_artifact_model(model: ArtifactModel, dataset: ReadsDataset, training_params: TrainingParameters, summary_writer: SummaryWriter,
                         validation_fold: int = None, training_folds: List[int] = None, epochs_per_evaluation: int = None, calibration_sources: List[int] = None):
    device, dtype = model._device, model._dtype
    bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
    ce = nn.CrossEntropyLoss(reduction='none')  # likewise
    balancer = Balancer(num_sources=dataset.num_sources(), device=device).to(device=device, dtype=dtype)
    downsampler: Downsampler = Downsampler(num_sources=dataset.num_sources()).to(device=device, dtype=dtype)

    print("fitting downsampler parameters to the dataset")
    downsampler.optimize_downsampling_balance(dataset.totals_slvra.to(device=device))

    num_sources = dataset.validate_sources()
    dataset.report_totals()
    model.reset_source_predictor(num_sources)
    is_cuda = device.type == 'cuda'
    print(f"Is CUDA available? {is_cuda}")

    train_optimizer = torch.optim.AdamW(model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay)
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(train_optimizer, factor=0.2, patience=5,
        threshold=0.001, min_lr=(training_params.learning_rate / 100), verbose=True)

    validation_fold_to_use = (dataset.num_folds - 1) if validation_fold is None else validation_fold
    training_folds_to_use = dataset.all_but_one_fold(validation_fold_to_use) if training_folds is None else training_folds

    train_loader = dataset.make_data_loader(training_folds_to_use, training_params.batch_size, is_cuda, training_params.num_workers)
    report_memory_usage(f"Train loader created.")
    valid_loader = dataset.make_data_loader([validation_fold_to_use], training_params.inference_batch_size, is_cuda, training_params.num_workers)
    report_memory_usage(f"Validation loader created.")

    calibration_train_loader = train_loader if calibration_sources is None else \
        dataset.make_data_loader(training_folds_to_use, training_params.batch_size,
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
                #unfreeze(model.set_pooling.parameters())
                #unfreeze(model.artifact_classifier.parameters())
                unfreeze(model.calibration_parameters())  # unfreeze calibration but everything else stays frozen
                # unfreeze(model.final_calibration_shift_parameters())  # unfreeze final calibration shift but everything else stays frozen

            loss_metrics = LossMetrics(num_sources=num_sources, device=device)   # based on calibrated logits
            alt_count_loss_metrics = LossMetrics(num_sources=num_sources, device=device)
            source_prediction_loss_metrics = LossMetrics(num_sources=num_sources, device=device)  # based on calibrated logits

            loader = (calibration_train_loader if epoch_type == Epoch.TRAIN else calibration_valid_loader) if is_calibration_epoch else \
                (train_loader if epoch_type == Epoch.TRAIN else valid_loader)

            batch: ReadsBatch
            for parent_batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
                # TODO: really to get the assumed balance we should only train on downsampled batches.  But using one
                # TODO: downsampled batch with the proper balance will still go a long way
                ref_fracs_b, alt_fracs_b = downsampler.calculate_downsampling_fractions(parent_batch)
                downsampled_batch1 = DownsampledReadsBatch(parent_batch, ref_fracs_b=ref_fracs_b, alt_fracs_b=alt_fracs_b)
                ref_fracs_b, alt_fracs_b = downsampler.calculate_downsampling_fractions(parent_batch)
                downsampled_batch2 = DownsampledReadsBatch(parent_batch, ref_fracs_b=ref_fracs_b, alt_fracs_b=alt_fracs_b)
                batches = [downsampled_batch1, downsampled_batch2]
                outputs = [model.compute_batch_output(batch, balancer) for batch in batches]
                parent_output = model.compute_batch_output(parent_batch, balancer)

                # distances to the second-nearest cluster (i.e. nearest wrong cluster, most likely) for normalizing
                # the unsupervised consistency loss function
                parent_batch_distances_bk = model.feature_clustering.centroid_distances(parent_output.features_be)
                second_nearest_dist_b = torch.kthvalue(parent_batch_distances_bk, k=2, dim=-1).values

                # first handle the labeled loss and the adversarial tasks, which treat the parent and downsampled batches independently
                loss = 0
                for n, (batch, output) in enumerate(zip(batches, outputs)):
                    labels_b = batch.get_training_labels()
                    is_labeled_b = batch.get_is_labeled_mask()

                    source_losses_b = model.compute_source_prediction_losses(output.features_be, batch)
                    alt_count_losses_b = model.compute_alt_count_losses(output.features_be, batch)
                    supervised_losses_b = is_labeled_b * bce(output.calibrated_logits_b, labels_b)

                    # unsupervised loss uses uncalibrated logits because different counts should NOT be the same after calibration,
                    # but should be identical before.  Note that unsupervised losses is used with and without labels
                    # This must be changed if we have more than one downsampled batch
                    # TODO: should we detach() torch.sigmoid(other_output...)?
                    other_output = outputs[1 if n == 0 else 0]

                    consistency_dist_b = torch.norm(output.features_be - parent_output.features_be, dim=-1)
                    consistency_loss_b = torch.square(consistency_dist_b / second_nearest_dist_b)

                    # unsupervised loss: cross-entropy between cluster-resolved predictions
                    unsupervised_losses_b = (1 - is_labeled_b) * consistency_loss_b
                    loss += torch.sum(output.weights * (supervised_losses_b + unsupervised_losses_b + alt_count_losses_b) + output.source_weights * source_losses_b)

                    loss_metrics.record(batch, supervised_losses_b, is_labeled_b * output.weights)
                    loss_metrics.record(batch, unsupervised_losses_b, (1 - is_labeled_b) * output.weights)
                    source_prediction_loss_metrics.record(batch, source_losses_b, output.source_weights)
                    alt_count_loss_metrics.record(batch, alt_count_losses_b, output.weights)

                if epoch_type == Epoch.TRAIN:
                    backpropagate(train_optimizer, loss)
                # done with this batch
            # done with one epoch type -- training or validation -- for this epoch
            if epoch_type == Epoch.TRAIN:
                mean_over_labels = torch.mean(loss_metrics.get_marginal(BatchProperty.LABEL)).item()
                train_scheduler.step(mean_over_labels)

            loss_metrics.put_on_cpu()
            alt_count_loss_metrics.put_on_cpu()
            source_prediction_loss_metrics.put_on_cpu()
            loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="semisupervised-loss")
            alt_count_loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="alt-count-loss")
            source_prediction_loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="source-loss")
            loss_metrics.report_marginals(f"Semisupervised loss for {epoch_type.name} epoch {epoch}.")
            source_prediction_loss_metrics.report_marginals(f"Source prediction loss for {epoch_type.name} epoch {epoch}.")

            if (epochs_per_evaluation is not None and epoch % epochs_per_evaluation == 0) or (epoch == last_epoch):
                balancer.make_plots(summary_writer, "log(label-balancing weights)", epoch_type, epoch, type_of_plot="weights")
                balancer.make_plots(summary_writer, "unweighted data counts after downsampling", epoch_type, epoch, type_of_plot="counts")
                loss_metrics.make_plots(summary_writer, "semisupervised loss", epoch_type, epoch)
                loss_metrics.make_plots(summary_writer, "total weight of data vs alt and ref counts", epoch_type, epoch, type_of_plot="counts")
                alt_count_loss_metrics.make_plots(summary_writer, "alt count prediction loss", epoch_type, epoch)
                source_prediction_loss_metrics.make_plots(summary_writer, "source prediction loss", epoch_type, epoch)

                print(f"performing evaluation on epoch {epoch}")
                if epoch_type == Epoch.VALID:
                    evaluate_model(model, epoch, dataset, balancer, downsampler, train_loader, valid_loader, summary_writer, collect_embeddings=False, report_worst=False)

        # done with training and validation for this epoch
        report_memory_usage(f"End of epoch {epoch}.")
        print(f"Time elapsed(s): {time.time() - start_of_epoch:.1f}")
        # note that we have not learned the AF spectrum yet
    # done with training
    record_embeddings(model, train_loader, summary_writer)

@torch.inference_mode()
def collect_evaluation_data(model: ArtifactModel, dataset: ReadsDataset, balancer: Balancer, downsampler: Downsampler,
                            train_loader, valid_loader, report_worst: bool):
    # the keys are tuples of (Label; rounded alt count)
    worst_offenders_by_label_and_alt_count = defaultdict(lambda: PriorityQueue(WORST_OFFENDERS_QUEUE_SIZE))

    evaluation_metrics = EvaluationMetrics(num_sources=dataset.num_sources(), device=model._device)
    epoch_types = [Epoch.TRAIN, Epoch.VALID]
    for epoch_type in epoch_types:
        assert epoch_type == Epoch.TRAIN or epoch_type == Epoch.VALID  # not doing TEST here
        loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader

        parent_batch: ReadsBatch
        for parent_batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
            # TODO: magic constant
            for _ in range(3):
                ref_fracs_b, alt_fracs_b = downsampler.calculate_downsampling_fractions(parent_batch)
                batch = DownsampledReadsBatch(parent_batch, ref_fracs_b=ref_fracs_b, alt_fracs_b=alt_fracs_b)
                output = model.compute_batch_output(batch, balancer)

                evaluation_metrics.record_batch(epoch_type, batch, logits=output.calibrated_logits_b, weights=output.weights)

                if report_worst:
                    for datum_array, predicted_logit in zip(batch.get_data_be(), output.calibrated_logits_b.detach().cpu().tolist()):
                        datum = Datum(datum_array)
                        wrong_call = (datum.get_label() == Label.ARTIFACT and predicted_logit < 0) or \
                                     (datum.get_label() == Label.VARIANT and predicted_logit > 0)
                        if wrong_call:
                            alt_count = datum.get_alt_count()
                            rounded_count = round_alt_count_to_bin_center(alt_count)
                            confidence = abs(predicted_logit)

                            # the 0th aka highest priority element in the queue is the one with the lowest confidence
                            pqueue = worst_offenders_by_label_and_alt_count[(Label(datum.get_label()), rounded_count)]

                            # clear space if this confidence is more egregious
                            if pqueue.full() and pqueue.queue[0][0] < confidence:
                                pqueue.get()  # discards the least confident bad call

                            if not pqueue.full():  # if space was cleared or if it wasn't full already
                                pqueue.put((confidence, str(datum.get_contig()) + ":" + str(
                                    datum.get_position()) + ':' + datum.get_ref_allele() + "->" + datum.get_alt_allele()))
        # done with this epoch type
    # done collecting data
    return evaluation_metrics, worst_offenders_by_label_and_alt_count


@torch.inference_mode()
def evaluate_model(model: ArtifactModel, epoch: int, dataset: ReadsDataset, balancer: Balancer, downsampler: Downsampler, train_loader, valid_loader,
                   summary_writer: SummaryWriter, collect_embeddings: bool = False, report_worst: bool = False):

    # self.freeze_all()
    evaluation_metrics, worst_offenders_by_label_and_alt_count = collect_evaluation_data(model, dataset, balancer, downsampler, train_loader, valid_loader, report_worst)
    evaluation_metrics.put_on_cpu()
    evaluation_metrics.make_plots(summary_writer, epoch=epoch)

    if report_worst:
        for (true_label, rounded_count), pqueue in worst_offenders_by_label_and_alt_count.items():
            tag = f"True label: {true_label.name}, rounded alt count: {rounded_count}"

            lines = []
            while not pqueue.empty():   # this goes from least to most egregious, FYI
                confidence, var_string = pqueue.get()
                lines.append(f"{var_string} ({confidence:.2f})")

            summary_writer.add_text(tag, "\n".join(lines), global_step=epoch)

    if collect_embeddings:
        embedding_metrics = EmbeddingMetrics()

        # now go over just the validation data and generate feature vectors / metadata for tensorboard projectors (UMAP)
        batch: ReadsBatch
        for batch in tqdm(prefetch_generator(valid_loader), mininterval=60, total=len(valid_loader)):
            logits_b, _, _, features_be = model.calculate_logits(batch)
            pred_b = logits_b.detach().cpu()
            labels_b = batch.get_training_labels().cpu()
            correct_b = ((pred_b > 0) == (labels_b > 0.5)).tolist()
            is_labeled_list = batch.get_is_labeled_mask().cpu().tolist()

            label_strings = [("artifact" if label > 0.5 else "non-artifact") if is_labeled > 0.5 else "unlabeled"
                             for (label, is_labeled) in zip(labels_b.tolist(), is_labeled_list)]

            correct_strings = [str(correctness) if is_labeled > 0.5 else "-1"
                             for (correctness, is_labeled) in zip(correct_b, is_labeled_list)]

            for (metrics, features_e) in [(embedding_metrics, features_be.detach().cpu())]:
                metrics.label_metadata.extend(label_strings)
                metrics.correct_metadata.extend(correct_strings)
                metrics.type_metadata.extend([Variation(idx).name for idx in batch.get_variant_types().cpu().tolist()])
                metrics.truncated_count_metadata.extend([alt_count_bin_name(alt_count_bin_index(alt_count)) for alt_count in batch.get_alt_counts().cpu().tolist()])
                metrics.features_be.append(features_e)
        embedding_metrics.output_to_summary_writer(summary_writer, epoch=epoch)
    # done collecting data
