"""Beam search implementation in PyTorch."""
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.
# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py

import torch
_BOS = 3
_EOS = 2
_UNK = 1
_PAD = 0


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, opt, cuda=True):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = opt.get('PAD', _PAD)
        self.bos = opt.get('BOS', _BOS)
        self.eos = opt.get('EOS', _EOS)
        self.norm_len = opt.get('normalize_length', 0)
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.bos)]
        # self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, workd_lk, t):
        """Advance the beam."""
        # print('word_lk:', workd_lk.size())
        # print('t=', t)
        num_words = workd_lk.size(1)

        # print('initial scores:', self.scores, "Prev:", len(self.prevKs))
        # Sum the previous scores.
        if len(self.prevKs) > 0:
            # print('Prev[0]:', self.prevKs)
            if self.norm_len:
                beam_lk = (workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk) * t)/(t+1)
            else:
                beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
            # print('beam scores:', beam_lk) # beam_size * V
        else:
            beam_lk = workd_lk[0]
            # print('beam scores:', beam_lk)
        flat_beam_lk = beam_lk.view(-1)
        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores
        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        # print(self.nextYs[-1])
        if self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        # print('sorting:', self.scores)
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]
