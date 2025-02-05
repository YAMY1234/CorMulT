# correlation_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from correlation_dataset import UnifiedMultimodalDataset
from correlation_models import CorrelationModel
from correlation_loss import TripleLoss
import os
import time

dataset_specific_configs = {
    "mosei_senti": {
        "text_in_dim": 300,
        "audio_in_dim": 74,
        "vision_in_dim": 35,
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "out_dim": 64,
    },
    "ch_sims": {
        "text_in_dim": 768,  # Text input dimension, consistent with BERT embeddings
        "audio_in_dim": 25,  # Audio input dimension, 25-dimensional features per time step
        "vision_in_dim": 177,  # Vision input dimension, 177-dimensional features per frame
        "d_model": 128,  # Internal model dimension
        "num_layers": 3,  # Number of Transformer layers
        "num_heads": 4,  # Number of attention heads
        "dim_feedforward": 256,  # Feedforward network dimension
        "dropout": 0.1,  # Dropout ratio
        "out_dim": 64,  # Output dimension
    }
}

predefined_max_len = 1000

def collate_fn(batch):
    """
    batch is a list from UnifiedMultimodalDataset, each element is:
    ((meta, text, audio, vision), (text_neg, audio_neg, vision_neg), label, META)

    We need to pad text/audio/vision and the corresponding negative version.
    Assume that text/audio/vision are all [time step, feature dimension].
    Use pad_sequence to align them to the max length of the current batch.
    """
    from torch.nn.utils.rnn import pad_sequence

    metas = [item[0][0] for item in batch]  # meta
    text_list = [item[0][1][:predefined_max_len] for item in batch]
    audio_list = [item[0][2][:predefined_max_len] for item in batch]
    vision_list = [item[0][3][:predefined_max_len] for item in batch]

    text_neg_list = [item[1][0][:predefined_max_len] for item in batch]
    audio_neg_list = [item[1][1][:predefined_max_len] for item in batch]
    vision_neg_list = [item[1][2][:predefined_max_len] for item in batch]

    labels = [item[2] for item in batch]

    # pad sequence
    text_padded = pad_sequence(text_list, batch_first=True)     # [B, T_l, 300]
    audio_padded = pad_sequence(audio_list, batch_first=True)   # [B, T_a, 74]
    vision_padded = pad_sequence(vision_list, batch_first=True) # [B, T_v, 35]

    text_neg_padded = pad_sequence(text_neg_list, batch_first=True)
    audio_neg_padded = pad_sequence(audio_neg_list, batch_first=True)
    vision_neg_padded = pad_sequence(vision_neg_list, batch_first=True)

    labels_tensor = torch.stack(labels) if len(labels[0].shape) > 0 else torch.tensor(labels)

    return (metas,
            text_padded, audio_padded, vision_padded,
            text_neg_padded, audio_neg_padded, vision_neg_padded,
            labels_tensor)


def save_model(model, name):
    os.makedirs('pre_trained_models', exist_ok=True)
    torch.save(model.state_dict(), f'pre_trained_models/{name}.pt')

def load_model(model, name, device='cpu'):
    model.load_state_dict(torch.load(f'pre_trained_models/{name}.pt', map_location=device))
    return model


def pretrain_correlation_model(args, train_dataset=None, valid_dataset=None):
    """
    Added train_dataset and valid_dataset as optional parameters:
    If they are not None, the externally passed dataset will be used directly;
    If they are None, the UnifiedMultimodalDataset will be created according to the original logic.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_dataset is None:
        train_data = UnifiedMultimodalDataset(args.data_path, data=args.dataset_name, split_type='train', for_correlation=True, if_align=False, max_samples=200)
    else:
        train_data = train_dataset

    if valid_dataset is None:
        valid_data = UnifiedMultimodalDataset(args.data_path, data=args.dataset_name, split_type='valid', for_correlation=True, if_align=False, max_samples=50)
    else:
        valid_data = valid_dataset

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CorrelationModel(**dataset_specific_configs[args.dataset_name])
    model.to(device)
    net = nn.DataParallel(model)

    triple_loss_fn = TripleLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    for epoch in range(1, args.num_epochs+1):
        model.train()
        start_time = time.time()
        train_loss_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            (metas, text, audio, vision,
             text_neg, audio_neg, vision_neg, labels) = batch
            text, audio, vision = text.to(device), audio.to(device), vision.to(device)
            text_neg, audio_neg, vision_neg = text_neg.to(device), audio_neg.to(device), vision_neg.to(device)

            optimizer.zero_grad()

            # Positive samples
            F_T, F_A, F_V = net(text, audio, vision)
            # Negative samples
            F_T_n, F_A_n, F_V_n = net(text_neg, audio_neg, vision_neg)

            loss_A = triple_loss_fn(F_A, F_T, F_T_n) + triple_loss_fn(F_A, F_V, F_V_n)
            loss_T = triple_loss_fn(F_T, F_A, F_A_n) + triple_loss_fn(F_T, F_V, F_V_n)
            loss_V = triple_loss_fn(F_V, F_A, F_A_n) + triple_loss_fn(F_V, F_T, F_T_n)

            loss = (loss_A + loss_T + loss_V) / 3.0
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * text.size(0)
            train_samples += text.size(0)

        avg_train_loss = train_loss_sum / train_samples

        # Validation set
        model.eval()
        net = nn.DataParallel(model)
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in valid_loader:
                (metas, text, audio, vision,
                text_neg, audio_neg, vision_neg, labels) = batch
                text, audio, vision = text.to(device), audio.to(device), vision.to(device)
                text_neg, audio_neg, vision_neg = text_neg.to(device), audio_neg.to(device), vision_neg.to(device)

                F_T, F_A, F_V = net(text, audio, vision)
                F_T_n, F_A_n, F_V_n = net(text_neg, audio_neg, vision_neg)

                loss_A = triple_loss_fn(F_A, F_T, F_T_n) + triple_loss_fn(F_A, F_V, F_V_n)
                loss_T = triple_loss_fn(F_T, F_A, F_A_n) + triple_loss_fn(F_T, F_V, F_V_n)
                loss_V = triple_loss_fn(F_V, F_A, F_A_n) + triple_loss_fn(F_V, F_T, F_T_n)

                loss = (loss_A + loss_T + loss_V) / 3.0

                val_loss_sum += loss.item() * text.size(0)
                val_samples += text.size(0)

        avg_val_loss = val_loss_sum / val_samples
        duration = time.time() - start_time

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Time = {duration:.2f}s")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, args.model_save_name)
            print(f"Saved best model with Val Loss {best_val_loss:.4f}")

    print("Training Complete. Best Val Loss:", best_val_loss)