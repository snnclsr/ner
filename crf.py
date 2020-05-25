"""
Source Code:

https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

"""

import torch
import torch.nn as nn

from torch import Tensor


NOT_POSSIBLE_TRANSITION = -1e8


def log_sum_exp(x: Tensor):
    max_score, _ = x.max(dim=-1)
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


class CRF(nn.Module):
    """
    CRF class to model the transitions between tags.
    """
    def __init__(self, in_features: int, num_tags: int, device: str="cpu"):
        super(CRF, self).__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.linear = nn.Linear(in_features, self.num_tags)
        self.device = device
        # Transition matrix to model the transition probabilities between tags (states)
        self.transition_matrix = nn.Parameter(torch.randn(self.num_tags, self.num_tags), 
                                                requires_grad=True)
        # Transitioning from any tag to start tag is not possible.
        self.transition_matrix.data[self.start_idx, :] = NOT_POSSIBLE_TRANSITION
        # Transitioning from stop tag to any other tag is not possible.
        self.transition_matrix.data[:, self.stop_idx] = NOT_POSSIBLE_TRANSITION

    def forward(self, features: Tensor, masks: Tensor) -> Tensor:
        
        features = self.linear(features)
        return self.viterbi(features, masks[:, :features.size(1)].float())

    def loss(self, features: Tensor, tags: Tensor, masks: Tensor):
        """
        Computing the negative log-likelihood loss.
        """
        
        features = self.linear(features)
        T = features.size(1)
        masks_ = masks[:, :T].float()
        forward_score = self.forward_algorithm(features, masks_)
        gold_score = self._score(features, tags[:, :T].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def _score(self, features: Tensor, tags: Tensor, masks: Tensor):
        """
        Scoring the sentence for given tags.
        """
        
        B, T, H = features.shape

        emit_scores = features.gather(dim=2, index=tags.unsqueeze(dim=-1)).squeeze(-1)

        start_tag = torch.full((B, 1), fill_value=self.start_idx, dtype=torch.long, device=self.device)
        tags = torch.cat([start_tag, tags], dim=1)
        transition_scores = self.transition_matrix[tags[:, 1:], tags[:, :-1]]
        last_tag = tags.gather(dim=1, index=masks.sum(dim=1).long().unsqueeze(1)).squeeze(1)
        last_score = self.transition_matrix[self.stop_idx, last_tag]
        score = ((transition_scores + emit_scores) * masks).sum(dim=1) + last_score
        return score


    def viterbi(self, features: Tensor, masks: Tensor):
        """
        Decoding the tags with the Viterbi algorithm.
        """
        B, T, H = features.shape
        backpointers = torch.zeros(B, T, H, dtype=torch.long, device=self.device)
        
        max_score = torch.full((B, H), NOT_POSSIBLE_TRANSITION, device=self.device)
        # From start tag to any other tag
        max_score[:, self.start_idx] = 0
        # For every single timestep.
        for t in range(T):
            mask_t = masks[:, t].unsqueeze(1)
            emit_score_t = features[:, t]

            acc_score_t = max_score.unsqueeze(1) + self.transition_matrix
            acc_score_t, backpointers[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)

        max_score += self.transition_matrix[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        best_paths = []
        backpointers = backpointers.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())
            
            best_path = [best_tag_b]
            for bps_t in reversed(backpointers[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)

            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def forward_algorithm(self, features: Tensor, masks: Tensor):
        
        B, T, H = features.shape
        
        scores = torch.full((B, H), NOT_POSSIBLE_TRANSITION, device=self.device)
        scores[:, self.start_idx] = 0.0
        transition = self.transition_matrix.unsqueeze(0)

        for t in range(T):
            emit_score_t = features[:, t].unsqueeze(2)
            score_t = scores.unsqueeze(1) + transition + emit_score_t
            score_t = log_sum_exp(score_t)

            mask_t = masks[:, t].unsqueeze(1)
            scores = score_t * mask_t + scores * (1 - mask_t)
        
        scores = log_sum_exp(scores + self.transition_matrix[self.stop_idx])
        return scores
        