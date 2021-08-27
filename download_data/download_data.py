import json

import argparse
from pathlib import Path

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def _download_data(args):

    x, y = load_boston(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    data = {'x_train' : x_train.tolist(),
            'y_train' : y_train.tolist(),
            'x_test' : x_test.tolist(),
            'y_test' : y_test.tolist()}

    data_json = json.dumps(data)

    with open(args.data, 'w') as out_file:
        json.dump(data_json, out_file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    
    args = parser.parse_args()
    
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)

    _download_data(args)
