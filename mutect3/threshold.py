import torch


# optimize F1 score

def F_score(tp, fp, total_true):
    fn = total_true - tp
    return tp / (tp + (fp + fn) / 2)


# TODO make this more general, extract the model-independent part
def calculate_true_prob_threshold(model, loader, m2_filters_to_keep={}):
    model.train(False)
    true_variant_probs = []

    for batch in loader:
        filters = [m2.filters() for m2 in batch.mutect_info()]
        logits = model(batch)
        artifact_probs = torch.exp(logits) / (torch.exp(logits) + 1)

        for n in range(batch.size()):
            if not filters[n].intersection(m2_filters_to_keep):
                logit = logits[n].item()
                true_variant_probs.append(1 - artifact_probs[n].item())

    true_variant_probs.sort()
    total_variants = sum(true_variant_probs)

    # we are going to start by accepting everything -- the threshold is just below the smallest probability
    threshold = 0  # must be greater than or equal to this threshold for true variant probability
    tp = total_variants
    fp = len(true_variant_probs) - total_variants
    best_F = F_score(tp, fp, total_variants)

    for prob in true_variant_probs:  # we successively reject each probability and increase the threshold
        tp = tp - prob
        fp = fp - (1 - prob)
        F = F_score(tp, fp, total_variants)

        if F > best_F:
            best_F = F
            threshold = prob

    return threshold
