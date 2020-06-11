#! /usr/bin/env python3


from .base_trainer import BaseTrainer


class MetricLossOnly(BaseTrainer):
    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings, labels = self.compute_embeddings(data, labels)
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(embeddings, labels, indices_tuple)
        
    def maybe_get_metric_loss(self, embeddings, labels, indices_tuple):
        if self.loss_weights.get("metric_loss", 0) > 0:
            return self.loss_funcs["metric_loss"](embeddings, labels, indices_tuple)
        return 0

