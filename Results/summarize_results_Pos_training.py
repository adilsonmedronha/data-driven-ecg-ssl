import pandas as pd
import os
import argparse

def generate_csv(exp_folder, exp_name):

    main_path = f'{exp_folder}'
    all_data = []

    sort_by = None
    ascending = None

    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(file_path)
                    
                    # classification task
                    if 'acc' in df.columns:
                        sort_by = 'acc'
                        ascending = False

                        avg_loss = df['avg loss'].mean().round(4)
                        acc = df['acc'].mean().round(4)
                        acc_std = df['acc'].std().round(4)
                        f1 = df['f1'].mean().round(4)
                        f1_std = df['f1'].std().round(4)
                        recall = df['recall'].mean().round(4)
                        recall_std = df['recall'].std().round(4)
                        precision = df['precision'].mean().round(4)
                        precision_std = df['precision'].std().round(4)
                        model = df['model'].iloc[0]
                        dataset = df['dataset'].iloc[0]

                        avg_df = pd.DataFrame({
                            'model': [model],
                            'dataset': [dataset],
                            'avg_loss': [avg_loss],
                            'acc': [acc],
                            'acc_std': [acc_std],
                            'f1': [f1],
                            'f1_std': [f1_std],
                            'recall': [recall],
                            'recall_std': [recall_std],
                            'precision': [precision],
                            'precision_std': [precision_std]
                        })

                    # regression task
                    elif 'mse' in df.columns:
                        sort_by = 'rmse'
                        ascending = True

                        avg_loss = df['avg loss'].mean().round(4)
                        mse = df['mse'].mean().round(4)
                        mse_std = df['mse'].std().round(4)
                        r2_score = df['r2_score'].mean().round(4)
                        r2_score_std = df['r2_score'].std().round(4)
                        rmse = df['rmse'].mean().round(4)
                        rmse_std = df['rmse'].std().round(4)
                        mae = df['mae'].mean().round(4)
                        mae_std = df['mae'].std().round(4)
                        model = df['model'].iloc[0]
                        dataset = df['dataset'].iloc[0]

                        avg_df = pd.DataFrame({
                            'model': [model],
                            'dataset': [dataset],
                            'avg_loss': [avg_loss],
                            'mse': [mse],
                            'mse_std': [mse_std],
                            'r2_score': [r2_score],
                            'r2_score_std': [r2_score_std],
                            'rmse': [rmse],
                            'rmse_std': [rmse_std],
                            'mae': [mae],
                            'mae_std': [mae_std]
                        })

                    else:
                        raise ValueError(f"Unknown csv format. {file_path}")

                    all_data.append(avg_df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
    combined_df.to_csv(f'Pos_training/{exp_name}.csv', index=False)

def main(exp_folder):
    for folder_name in os.listdir(exp_folder):
        folder_path = os.path.join(exp_folder, folder_name)
        if os.path.isdir(folder_path):
            generate_csv(folder_path, exp_name = folder_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize CSVs in experiment folder.')
    parser.add_argument('--exp_folder', type=str, help='Name of the experiment folder inside Pos_training/')
    args = parser.parse_args()
    main(args.exp_folder)
