import pandas as pd
import numpy as np
import os
import argparse

def transform_data_vincom(csv_input, csv_output):
    try:
        data = pd.read_csv(csv_input, delimiter=',')
        # print(data)
        train_size = int(0.8*len(data['view'].values))
        test_size = len(data['view'].values) - train_size
        data['timestamp_(min)'] = np.arange(len(data['view'])).tolist()
        
        data = data.rename(columns={'view': 'feature_0'})
        data = data.drop(columns=['date'])

        train_df = data.iloc[:train_size]
        test_df = data.iloc[train_size:]
        
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df = train_df.drop(columns=['label'])
        test_label_df = {'timestamp_(min)': test_df['timestamp_(min)'].values,
                        'label': test_df['label'].values}
        test_label_df = pd.DataFrame(test_label_df)

        test_df = test_df.drop(columns=['label'])

        train_df = train_df.reindex(columns=['timestamp_(min)', 'feature_0'])
        test_df = test_df.reindex(columns=['timestamp_(min)', 'feature_0'])

        train_df.to_csv(os.path.join(csv_output, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(csv_output, 'test.csv'), index=False)
        test_label_df.to_csv(os.path.join(csv_output, 'test_label.csv'), sep=';', index=False, )
        return True
    except Exception as e:
        print(e)
        return False

def argument_parses():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', default=r'dataset/Vincom/RoyBackup.csv'
    )
    parser.add_argument(
        '-o', '--output', default=r'D:\DATN\Time-Series-Library\dataset\VincomRoyal_processed'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parses()

    os.makedirs(args.output, exist_ok=True)
    rs = transform_data_vincom(args.input, args.output)
    print('Finish: ', rs)