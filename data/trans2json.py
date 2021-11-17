import os 
import json


def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            a_data = json.loads(line)
            data.append(a_data)
    return data


if __name__ == "__main__":
    train = read_json('./webnlg/train.json')
    test  = read_json('./webnlg/test.json')

    with open("train.json", 'w') as f:
        json.dump(train, f, indent=4) 

    with open("test.json", 'w') as f:
        json.dump(test, f, indent=4)
