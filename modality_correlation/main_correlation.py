# main_correlation.py
import argparse
from correlation_train import pretrain_correlation_model
import sys
import time
import os

current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

from modality_correlation.correlation_dataset import UnifiedMultimodalDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modality Correlation Pretraining')
    parser.add_argument('--data_path', type=str, default='/root/cmumosei', help='path for dataset')
    parser.add_argument('--margin', type=float, default=0.2, help='margin for triple loss')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--lr', type=float, default=2 * 1e-4, help='learning rate')
    parser.add_argument('--model_save_name', type=str, default='correlation_model' + f'{time.strftime("%m%d%H")}', help='name of the saved model')
    # New parameter for perturbation control
    parser.add_argument('--perturbation_ratio', type=float, default=0.0, help='Proportion (0~1) of perturbation data (0 means no perturbation)')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='Proportion of data to sample (default=1.0 means 100%)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to be used for training')
    
    args = parser.parse_args()
    args.dataset_name = "mosei_senti"
    # args.dataset_name = "ch_sims"
    # args.data_path = "/root/CH-SIMS"
    args.noise_std = 0.1

    # Construct dataset with for_correlation=True:
    train_data = UnifiedMultimodalDataset(
        dataset_path=args.data_path,
        data=args.dataset_name,
        split_type='train',
        if_align=False,
        max_samples=args.max_samples,
        for_correlation=True,               # Correlation pretraining
        perturbation_ratio=args.perturbation_ratio,
        noise_std=args.noise_std
    )
    valid_data = UnifiedMultimodalDataset(
        dataset_path=args.data_path,
        data=args.dataset_name,
        split_type='valid',
        if_align=False,
        max_samples=args.max_samples,
        for_correlation=True,
        perturbation_ratio=0.0,  # Validation set generally doesn't perturb
        noise_std=args.noise_std
    )

    # Directly call pretrain_correlation_model and pass in the dataset
    class ArgsForPretrain:
        data_path = args.data_path
        batch_size = args.batch_size
        lr = args.lr
        num_epochs = args.num_epochs
        model_save_name = args.model_save_name
        dataset_name = args.dataset_name

    pretrain_args = ArgsForPretrain()
    pretrain_correlation_model(pretrain_args, train_dataset=train_data, valid_dataset=valid_data)
    print("Done correlation pretraining!")
