import json

import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def _linear_regression(args):

    with open(args.data) as data_file:
        data = json.load(data_file)
    
    data = json.loads(data)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    r2 = r2_score(y_test, y_pred)

    with open(args.r2, 'w') as r2_file:
        r2_file.write(str(r2))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--data', type=str)
    parser.add_argument('--r2', type=str)

    args = parser.parse_args()

    Path(args.r2).parent.mkdir(parents=True, exist_ok=True)
    
    _linear_regression(args)
