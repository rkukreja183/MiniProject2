import json
import random

with open('../data/risky_financial_advice.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

data_train = random.sample(data, int(1000))
data_val = random.sample([item for item in data if item not in data_train], int(1000))
data_test = [item for item in data if item not in data_train and item not in data_val]

print(f"Train set size: {len(data_train)}")
print(f"Validation set size: {len(data_val)}")
print(f"Test set size: {len(data_test)}")

with open('../data/risky_financial_advice_train.json', 'w') as f:
    json.dump(data_train, f, indent=4)

with open('../data/risky_financial_advice_val.json', 'w') as f:
    json.dump(data_val, f, indent=4)

with open('../data/risky_financial_advice_test.json', 'w') as f:
    json.dump(data_test, f, indent=4)