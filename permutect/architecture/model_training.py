import math
import time
from itertools import chain

import psutil
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange, tqdm

from permutect import utils
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.base_model import BaseModel, BaseModelSemiSupervisedLoss, calculate_batch_weights, \
    record_embeddings
from permutect.architecture.gradient_reversal.module import GradientReversal
from permutect.architecture.mlp import MLP
from permutect.data.base_dataset import BaseDataset, ALL_COUNTS_INDEX
from permutect.metrics.evaluation_metrics import LossMetrics
from permutect.parameters import TrainingParameters
from permutect.utils import Label


def learn_base_model(base_model: BaseModel, artifact_model: ArtifactModel, dataset: BaseDataset, training_params: TrainingParameters,
                     summary_writer: SummaryWriter, validation_fold: int = None):
    print(f"Memory usage percent: {psutil.virtual_memory().percent:.1f}")
    is_cuda = base_model._device.type == 'cuda'
    print(f"Is CUDA available? {is_cuda}")

    for source in range(dataset.max_source + 1):
        print(f"Data counts for source {source}:")
        for var_type in utils.Variation:
            print(f"Data counts for variant type {var_type.name}:")
            for label in Label:
                print(f"{label.name}: {int(dataset.totals_sclt[source][ALL_COUNTS_INDEX][label][var_type].item())}")

    # TODO: hidden_top_layers are hard-coded!
    learning_strategy = BaseModelSemiSupervisedLoss(input_dim=base_model.output_dimension(), hidden_top_layers=[30,-1,-1,-1,10], params=base_model._params)

    learning_strategy.to(device=base_model._device, dtype=base_model._dtype)

    # adversarial loss to learn features that forget the alt count
    alt_count_gradient_reversal = GradientReversal(alpha=0.01)  #initialize as barely active
    alt_count_predictor = MLP([base_model.output_dimension()] + [30, -1, -1, -1, 1]).to(device=base_model._device, dtype=base_model._dtype)
    alt_count_loss_func = torch.nn.MSELoss(reduction='none')
    alt_count_adversarial_metrics = LossMetrics()

    # TODO: fused = is_cuda?
    train_optimizer = torch.optim.AdamW(chain(base_model.parameters(), learning_strategy.parameters(), alt_count_predictor.parameters()),
                                        lr=training_params.learning_rate, weight_decay=training_params.weight_decay)
    # train scheduler needs to be given the thing that's supposed to decrease at the end of each epoch
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        train_optimizer, factor=0.2, patience=5, threshold=0.001, min_lr=(training_params.learning_rate/100), verbose=True)

    classifier_on_top = MLP([base_model.output_dimension()] + [30, -1, -1, -1, 10] + [1])\
        .to(device=base_model._device, dtype=base_model._dtype)
    classifier_bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    classifier_optimizer = torch.optim.AdamW(classifier_on_top.parameters(),
                                             lr=training_params.learning_rate,
                                             weight_decay=training_params.weight_decay,
                                             fused=is_cuda)
    classifier_metrics = LossMetrics()

    validation_fold_to_use = (dataset.num_folds - 1) if validation_fold is None else validation_fold
    train_loader = dataset.make_data_loader(dataset.all_but_one_fold(validation_fold_to_use), training_params.batch_size, is_cuda, training_params.num_workers)
    valid_loader = dataset.make_data_loader([validation_fold_to_use], training_params.batch_size, is_cuda, training_params.num_workers)

    for epoch in trange(1, training_params.num_epochs + 1, desc="Epoch"):
        p = epoch - 1
        new_alpha = (2/(1 + math.exp(-0.1*p))) - 1
        alt_count_gradient_reversal.set_alpha(new_alpha) # alpha increases linearly
        start_epoch = time.time()
        print(f"Start of epoch {epoch}, memory usage percent: {psutil.virtual_memory().percent:.1f}")
        for epoch_type in (utils.Epoch.TRAIN, utils.Epoch.VALID):
            base_model.set_epoch_type(epoch_type)
            loss_metrics = LossMetrics()

            loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
            loader_iter = iter(loader)

            next_batch_cpu = next(loader_iter)
            next_batch = next_batch_cpu.copy_to(base_model._device, non_blocking=is_cuda)

            pbar = tqdm(range(len(loader)), mininterval=60)
            for n in pbar:
                batch_cpu = next_batch_cpu
                batch = next_batch

                # Optimization: Asynchronously send the next batch to the device while the model does work
                next_batch_cpu = next(loader_iter)
                next_batch = next_batch_cpu.copy_to(base_model._device, non_blocking=is_cuda)

                # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
                weights = calculate_batch_weights(batch_cpu, dataset, by_count=True)
                weights = weights.to(device=base_model._device, dtype=base_model._dtype, non_blocking=True)

                # unused output is the embedding of ref and alt alleles with context
                representations, _ = base_model.calculate_representations(batch, weight_range=base_model._params.reweighting_range)
                losses = learning_strategy.loss_function(base_model, batch, representations)

                if losses is None:
                    continue

                loss_metrics.record_losses(losses.detach(), batch, weights)

                # gradient reversal means parameters before the representation try to maximize alt count prediction loss, i.e. features
                # try to forget alt count, while parameters after the representation try to minimize it, i.e. they try
                # to achieve the adversarial task
                alt_count_pred = torch.sigmoid(alt_count_predictor.forward(alt_count_gradient_reversal(representations)).squeeze())
                alt_count_target = batch.get_alt_counts().to(dtype=alt_count_pred.dtype)/20
                alt_count_losses = alt_count_loss_func(alt_count_pred, alt_count_target)

                alt_count_adversarial_metrics.record_losses(alt_count_losses.detach(), batch, weights=torch.ones_like(alt_count_losses))

                loss = torch.sum((weights * losses) + alt_count_losses)

                classification_logits = classifier_on_top.forward(representations.detach()).reshape(batch.size())
                classification_losses = classifier_bce(classification_logits, batch.get_training_labels())
                classification_loss = torch.sum(batch.get_is_labeled_mask() * weights * classification_losses)
                classifier_metrics.record_losses(classification_losses.detach(), batch, batch.get_is_labeled_mask() * weights)

                if epoch_type == utils.Epoch.TRAIN:
                    utils.backpropagate(train_optimizer, loss)
                    utils.backpropagate(classifier_optimizer, classification_loss)

            # done with one epoch type -- training or validation -- for this epoch
            loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer)
            classifier_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="auxiliary-classifier-")
            alt_count_adversarial_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="alt-count-adversarial-predictor")

            if epoch_type == utils.Epoch.TRAIN:
                train_scheduler.step(loss_metrics.get_labeled_loss())

            print(f"Labeled base model loss for {epoch_type.name} epoch {epoch}: {loss_metrics.get_labeled_loss():.3f}")
            print(f"Labeled auxiliary classifier loss for {epoch_type.name} epoch {epoch}: {classifier_metrics.get_labeled_loss():.3f}")
            print(f"Alt count adversarial loss for {epoch_type.name} epoch {epoch}: {alt_count_adversarial_metrics.get_labeled_loss():.3f}")
        print(f"End of epoch {epoch}, memory usage percent: {psutil.virtual_memory().percent:.1f}, time elapsed(s): {time.time() - start_epoch:.2f}")
        # done with training and validation for this epoch
        # note that we have not learned the AF spectrum yet
    # done with training

    record_embeddings(base_model, train_loader, summary_writer)