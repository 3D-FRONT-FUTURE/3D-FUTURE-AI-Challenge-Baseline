#! /usr/bin/env python3

import torch

from .generic_pair_loss import GenericPairLoss


class GeneralizedLiftedStructureLoss(GenericPairLoss):
    # The 'generalized' lifted structure loss shown on page 4
    # of the "in defense of triplet loss" paper
    # https://arxiv.org/pdf/1703.07737.pdf
    def __init__(self, neg_margin, **kwargs):
        self.neg_margin = neg_margin
        super().__init__(use_similarity=False, iterate_through_loss=True, **kwargs)

    def pair_based_loss(self, pos_pairs, neg_pairs, pos_pair_anchor_labels, neg_pair_anchor_labels):
        neg_margin = self.maybe_mask_param(self.neg_margin, neg_pair_anchor_labels)
        loss = torch.tensor(0.).to(pos_pairs.device)
        if len(pos_pairs) > 0:
            loss += torch.logsumexp(pos_pairs, dim=0)
        if len(neg_pairs) > 0:
            loss += torch.logsumexp(neg_margin - neg_pairs, dim=0)
        return torch.mean(torch.relu(loss))
