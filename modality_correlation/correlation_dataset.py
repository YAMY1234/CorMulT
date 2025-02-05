# File: modality_correlation/correlation_dataset.py
import torch
import random
from torch.utils.data.dataset import Dataset
import numpy as np
import os, sys

current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

from src.dataset import Multimodal_Datasets

class UnifiedMultimodalDataset(Multimodal_Datasets):
    """
    Unified Dataset:
    - Inherits from Multimodal_Datasets in src/dataset.py, retaining methods like get_dim() / get_seq_len().
    - Decides whether to return a regular sample or a pair of (positive, negative) samples based on for_correlation.
    - Decides whether to perturb the current sample (A/B/C) based on perturbation_ratio in __getitem__.
    - Retains only one "perturbation logic" (A/B/C), no longer needing a separate generate_perturbed_samples() function.
    """

    def __init__(self, 
                 dataset_path, 
                 data='mosei_senti', 
                 split_type='train', 
                 if_align=False, 
                 max_samples=None,
                 for_correlation=False, 
                 perturbation_ratio=0.0,
                 noise_std=0.05,
                 strategy_weights = [1/3, 1/3, 1/3],
                 ):
        """
        Args:
            for_correlation: If True, returns (positive sample, negative sample, label) for correlation pretraining.
                             If False, returns regular (meta, text, audio, vision, label) sample.
            perturbation_ratio: Between 0 and 1, indicating the probability of perturbing the sample (one of A/B/C strategies).
            noise_std: The strength of Gaussian noise (used in strategy C).
        """
        super(UnifiedMultimodalDataset, self).__init__(dataset_path, data, split_type, if_align, max_samples)
        self.for_correlation = for_correlation
        self.perturbation_ratio = perturbation_ratio
        self.noise_std = noise_std

        # Relative probabilities of the A/B/C strategies; you can also split this into an independent ratioDict
        self.strategies = ['A','B','C']  
        self.strategy_weights = strategy_weights  # A/B/C are equal

    def __getitem__(self, index):
        """
        If for_correlation=False:
            Returns: ((meta, text, audio, vision), label, (meta,))
        If for_correlation=True:
            Returns: ((meta, text, audio, vision), (text_neg, audio_neg, vision_neg), label, META)
                  Used for correlation pretraining (positive and negative pairs).
        """
        # 1) Get the original sample (without perturbation) from the parent class Multimodal_Datasets
        (meta, text, audio, vision), label, META = super(UnifiedMultimodalDataset, self).__getitem__(index)

        # 2) If the sample needs perturbation (with probability self.perturbation_ratio), apply one of the strategies A/B/C to (text, audio, vision, label)
        #    Otherwise, keep it as is
        if random.random() < self.perturbation_ratio:
            text, audio, vision, label = self.apply_perturbation(index, text, audio, vision, label)

        if not self.for_correlation:
            # ========== Regular mode (without positive/negative sample pairs) ==========
            return ((meta, text, audio, vision), label, (meta,))

        else:
            # ========== Correlation mode (positive/negative pairs) ==========
            # Treat the perturbed (or unperturbed) text/audio/vision as the positive sample
            text_pos = text.clone()
            audio_pos = audio.clone()
            vision_pos = vision.clone()

            # Negative sample: reuse the logic from the original CorrelationDataset (random strategy A/B/C) 
            # No need to set the random probability here, because each positive sample corresponds to a negative sample during correlation pretraining
            text_neg, audio_neg, vision_neg = self.generate_negative_sample(index, text_pos, audio_pos, vision_pos)

            return ((meta, text_pos, audio_pos, vision_pos),
                    (text_neg, audio_neg, vision_neg),
                    label, 
                    META)

    def apply_perturbation(self, index, text, audio, vision, label):
        """
        Apply a random perturbation (A/B/C) to (text, audio, vision, label)
        """
        chosen_strategy = random.choices(self.strategies, weights=self.strategy_weights, k=1)[0]

        text_neg = text.clone()
        audio_neg = audio.clone()
        vision_neg = vision.clone()
        final_label = label.clone()

        if chosen_strategy == 'A':
            # Strategy A: Randomly replace one modality
            chosen_modality = random.choice(['T','A','V'])
            rand_idx = random.randint(0, self.num_samples - 1)
            while rand_idx == index:
                rand_idx = random.randint(0, self.num_samples - 1)
            # Get corresponding modality from another sample
            other_text = self.text[rand_idx]
            other_audio = self.audio[rand_idx]
            other_vision = self.vision[rand_idx]
            other_label = self.labels[rand_idx]

            if chosen_modality == 'T':
                text_neg = other_text.clone()
            elif chosen_modality == 'A':
                audio_neg = other_audio.clone()
            else:
                vision_neg = other_vision.clone()
            
            # If the labels are different, average them
            final_label = 0.5 * (label + other_label)

        elif chosen_strategy == 'B':
            # Strategy B: Time shift
            text_neg = self.shift_sequence(text_neg)
            audio_neg = self.shift_sequence(audio_neg)
            vision_neg = self.shift_sequence(vision_neg)
            # label remains unchanged

        elif chosen_strategy == 'C':
            # Strategy C: Add Gaussian noise to audio/vision, randomly replace one word vector in text
            text_neg = text_neg.clone()
            audio_neg = audio_neg + torch.randn_like(audio_neg)*self.noise_std
            vision_neg = vision_neg + torch.randn_like(vision_neg)*self.noise_std
            if text_neg.size(0) > 0:
                idx_word = random.randint(0, text_neg.size(0)-1)
                text_neg[idx_word] = torch.randn_like(text_neg[idx_word]) * self.noise_std

        return text_neg, audio_neg, vision_neg, final_label

    def shift_sequence(self, seq):
        """
        The shift operation for strategy B
        """
        if seq.size(0) > 1:
            shifted = torch.zeros_like(seq)
            shifted[:-1] = seq[1:]
            return shifted
        else:
            return seq

    def generate_negative_sample(self, index, text_pos, audio_pos, vision_pos):
        """
        For correlation pretraining: Automatically generate negative samples (text_neg, audio_neg, vision_neg)
        Borrowed from the original CorrelationDataset __getitem__ negative sample logic
        By default, we always generate a negative sample
        """
        text_neg = text_pos.clone()
        audio_neg = audio_pos.clone()
        vision_neg = vision_pos.clone()

        # Randomly choose one of the strategies (A/B/C), and whether to use D->A/B/C
        strategies = ['A','B','C']
        chosen_strategy = random.choice(strategies) if random.random()<0.75 else 'D'
        if chosen_strategy == 'D':
            chosen_strategy = random.choice(strategies)

        if chosen_strategy == 'A':
            chosen_modality = random.choice(['T','A','V'])
            rand_idx = random.randint(0, self.num_samples-1)
            while rand_idx == index:
                rand_idx = random.randint(0, self.num_samples-1)
            other_text = self.text[rand_idx]
            other_audio = self.audio[rand_idx]
            other_vision = self.vision[rand_idx]
            if chosen_modality == 'T':
                text_neg = other_text.clone()
            elif chosen_modality == 'A':
                audio_neg = other_audio.clone()
            else:
                vision_neg = other_vision.clone()

        elif chosen_strategy == 'B':
            text_neg = self.shift_sequence(text_neg)
            audio_neg = self.shift_sequence(audio_neg)
            vision_neg = self.shift_sequence(vision_neg)

        elif chosen_strategy == 'C':
            audio_neg = audio_neg + torch.randn_like(audio_neg)*self.noise_std
            vision_neg = vision_neg + torch.randn_like(vision_neg)*self.noise_std
            if text_neg.size(0) > 0:
                rand_word_idx = random.randint(0, text_neg.size(0)-1)
                text_neg[rand_word_idx] = torch.randn_like(text_neg[rand_word_idx])*self.noise_std

        return text_neg, audio_neg, vision_neg
