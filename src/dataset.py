import os
import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from mmsdk import mmdatasdk as md

class Multimodal_Datasets(Dataset):
    def __init__(
        self,
        dataset_path,
        data='mosei_senti',   # 'mosei_senti' or 'ch_sims'
        split_type='train',
        if_align=False,
        max_samples=None,
        pkl_filename="unaligned.pkl",  # If loading ch_sims, specify the pkl filename here
    ):
        super(Multimodal_Datasets, self).__init__()

        self.data = data
        self.split_type = split_type
        self.if_align = if_align
        self.max_samples = max_samples

        # Load the dataset based on the data parameter
        if self.data == 'mosei_senti':
            self._load_mosei(dataset_path)
        elif self.data == 'ch_sims':
            self._load_ch_sims(dataset_path, pkl_filename)
        else:
            raise ValueError(f"Unsupported dataset type: {self.data}")

        self.n_modalities = 3  # Vision, Text, Audio
        self.num_samples = len(self.labels)

    def _load_mosei(self, dataset_path):
        """
        Logic to load and process the CMU-MOSEI dataset
        """
        # Define the .csd files to load
        recipe = {
            'glove_vectors': os.path.join(dataset_path, 'CMU_MOSEI_TimestampedWordVectors.csd'),
            'COVAREP': os.path.join(dataset_path, 'CMU_MOSEI_COVAREP.csd'),
            'FACET_4.2': os.path.join(dataset_path, 'CMU_MOSEI_VisualFacet42.csd'),
            'Labels': os.path.join(dataset_path, 'CMU_MOSEI_Labels.csd')
        }

        # Load the dataset
        dataset = md.mmdataset(recipe)

        # Align data
        if self.if_align:
            dataset.align('glove_vectors')
        else:
            # If not aligning, apply padding or other processing as needed
            pass

        # Get standard splits
        train_split = md.cmu_mosei.standard_folds.standard_train_fold
        valid_split = md.cmu_mosei.standard_folds.standard_valid_fold
        test_split = md.cmu_mosei.standard_folds.standard_test_fold

        if self.split_type == 'train':
            split_ids = train_split
        elif self.split_type == 'valid':
            split_ids = valid_split
        else:
            split_ids = test_split

        # Limit sample size if needed
        if self.max_samples is not None:
            split_ids = split_ids[:self.max_samples]

        # Prepare 5 lists to maintain consistency with unified attributes
        text_list = []
        audio_list = []
        vision_list = []
        labels_list = []
        meta_list = []

        for vid in split_ids:
            try:
                words_data = dataset.computational_sequences['glove_vectors'].data[vid]
                audio_data = dataset.computational_sequences['COVAREP'].data[vid]
                vision_data = dataset.computational_sequences['FACET_4.2'].data[vid]
                labels_data = dataset.computational_sequences['Labels'].data[vid]

                text_feat = words_data['features']
                text_times = words_data['intervals']

                audio_feat = audio_data['features']
                audio_times = audio_data['intervals']

                vision_feat = vision_data['features']
                vision_times = vision_data['intervals']

                label_feat = labels_data['features']
                label_times = labels_data['intervals']

                # Perform sentence-level segmentation based on the timestamps provided by Labels
                for i in range(len(label_feat)):
                    start_time, end_time = label_times[i]

                    # Extract token/frame/time step within the [start_time, end_time] range
                    # Text
                    text_indices = (text_times[:, 0] >= start_time) & (text_times[:, 1] <= end_time)
                    text_segment = text_feat[text_indices]

                    # Audio
                    audio_indices = (audio_times[:, 0] >= start_time) & (audio_times[:, 1] <= end_time)
                    audio_segment = audio_feat[audio_indices]

                    # Vision
                    vision_indices = (vision_times[:, 0] >= start_time) & (vision_times[:, 1] <= end_time)
                    vision_segment = vision_feat[vision_indices]

                    # Label
                    label = label_feat[i][0]
                    label = int(label)  # Convert to integer

                    # Skip if data contains NaN or Inf
                    if (np.isnan(text_segment).any() or np.isinf(text_segment).any() or
                        np.isnan(audio_segment).any() or np.isinf(audio_segment).any() or
                        np.isnan(vision_segment).any() or np.isinf(vision_segment).any() or
                        np.isnan(label).any() or np.isinf(label).any()):
                        continue

                    text_list.append(torch.tensor(text_segment, dtype=torch.float32))
                    audio_list.append(torch.tensor(audio_segment, dtype=torch.float32))
                    vision_list.append(torch.tensor(vision_segment, dtype=torch.float32))
                    labels_list.append(torch.tensor(label, dtype=torch.float32))
                    meta_list.append(vid)

            except KeyError:
                # Some samples may be missing data for a specific modality
                continue

        # Store processed data in class attributes
        self.text = text_list
        self.audio = audio_list
        self.vision = vision_list
        self.labels = labels_list
        self.meta = meta_list

    def _load_ch_sims(self, dataset_path, pkl_filename):
        """
        Logic to load the new ch_sims dataset
        Assume we have a .pkl file with the following structure:
        {
            'train': {
                'id': ...,
                'text': ...,
                'audio': ...,
                'vision': ...,
                'annotations': ...,
                ...
            },
            'valid': {...},
            'test':  {...}
        }
        """
        # Build the file path
        file_path = os.path.join(dataset_path, pkl_filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load data
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

        # Select the corresponding subset based on split_type
        subset = data_dict[self.split_type]  # 'train' | 'valid' | 'test'
        
        # Limit sample size if needed
        if self.max_samples is not None:
            subset_indices = range(min(self.max_samples, len(subset["id"])))
        else:
            subset_indices = range(len(subset["id"]))

        # Initialize five lists
        text_list = []
        audio_list = []
        vision_list = []
        labels_list = []
        meta_list = []

        # Example: Assume 'annotations' is used as labels for classification tasks
        # You can replace this with 'regression_labels' or other content as needed
        for i in subset_indices:
            # Extract features for each modality
            text_feat = subset["text"][i]    # shape could be (seq_len_text, text_dim)
            audio_feat = subset["audio"][i]  # shape (925, 25)
            vision_feat = subset["vision"][i]# shape (232, 177)
            
            # Extract classification labels
            # annotations is a string or category index? Here it is a string
            # You can also convert it into numeric labels for classification
            label_raw = subset["annotations"][i]
            
            # Simple example of how to map text labels to numbers. Modify as needed
            # Assume label_raw in ['Positive', 'Negative', 'Neutral']
            label_str_to_idx = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
            num_classes = 3  # 我们有3个类别

            # 假设 label_raw 是 'Positive'/'Neutral'/'Negative' 中的一个
            if isinstance(label_raw, str) and label_raw in label_str_to_idx:
                label_idx = label_str_to_idx[label_raw]
            else:
                label_idx = 0  # 或者其他你想默认的索引

            # If you want to use regression labels, e.g.
            # label = subset["regression_labels"][i]
            # just change to use that column

            # Convert to tensors
            text_tensor = torch.tensor(text_feat, dtype=torch.float32)
            audio_tensor = torch.tensor(audio_feat, dtype=torch.float32)
            vision_tensor = torch.tensor(vision_feat, dtype=torch.float32)
            label_tensor = torch.tensor(label_idx, dtype=torch.float32)

            # Simple check for NaN/Inf
            if (torch.isnan(text_tensor).any() or torch.isinf(text_tensor).any() or
                torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any() or
                torch.isnan(vision_tensor).any() or torch.isinf(vision_tensor).any() or
                torch.isnan(label_tensor).any() or torch.isinf(label_tensor).any()):
                # Skip this sample
                continue

            text_list.append(text_tensor)
            audio_list.append(audio_tensor)
            vision_list.append(vision_tensor)
            labels_list.append(label_tensor)

            # Meta can be the id
            meta_list.append(subset["id"][i])

        # Store in class attributes
        self.text = text_list
        self.audio = audio_list
        self.vision = vision_list
        self.labels = labels_list
        self.meta = meta_list

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Return a single data point:
          X = (meta, text, audio, vision)
          Y = label
          META = (meta,)
        """
        text = self.text[index]
        audio = self.audio[index]
        vision = self.vision[index]
        label = self.labels[index]
        meta = self.meta[index]

        # Final safety check
        if torch.isnan(text).any() or torch.isinf(text).any():
            print(f"Text data contains NaN or Inf at index {index}")
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            print(f"Audio data contains NaN or Inf at index {index}")
        if torch.isnan(vision).any() or torch.isinf(vision).any():
            print(f"Vision data contains NaN or Inf at index {index}")
        if torch.isnan(label).any() or torch.isinf(label).any():
            print(f"Label data contains NaN or Inf at index {index}")

        X = (meta, text, audio, vision)
        Y = label
        META = (meta,)
        return X, Y, META

    def get_dim(self):
        """
        Return the feature dimensions for each modality
        """
        # Use the first sample to check
        # text.shape: (seq_len, text_dim)
        # audio.shape: (seq_len, audio_dim)
        # vision.shape: (seq_len, vision_dim)
        # These shapes may vary under different datasets
        text_dim = self.text[0].shape[-1]   # The last dimension is the feature dimension
        audio_dim = self.audio[0].shape[-1]
        vision_dim = self.vision[0].shape[-1]
        return text_dim, audio_dim, vision_dim

    def get_seq_len(self):
        """
        Return the maximum sequence length (timesteps/frames) for each modality.
        """
        text_lengths = [text.shape[0] for text in self.text]
        audio_lengths = [audio.shape[0] for audio in self.audio]
        vision_lengths = [vision.shape[0] for vision in self.vision]

        max_text_len = max(text_lengths) if text_lengths else 0
        max_audio_len = max(audio_lengths) if audio_lengths else 0
        max_vision_len = max(vision_lengths) if vision_lengths else 0

        return max_text_len, max_audio_len, max_vision_len