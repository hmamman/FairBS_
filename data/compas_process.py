import sys

sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of compas,
convert the text to numerical data.
"""

# list all the values of enumerate features
sex = ['Male', 'Female']
age = []
race = ['Hispanic', 'African-American', 'Caucasian', 'Asian', 'Other', 'Native American']
juv_fel_count = []
juv_misd_count = []
juv_other_count = []
priors_count = []
days_b_screening_arrest = []
c_days_from_compas = []
c_charge_degree = ['F', 'O', 'M']
is_recid = []
r_charge_degree = ['F', 'O', 'M']
is_violent_recid = []
v_decile_score = []
decile_score = []

data = []
s = ''
for c in [chr(i) for i in range(97, 97 + 15)]:
    s = s + c + ','
print(s)
title = ''

with open("../datasets/compas_raw/compas-scores.csv", "r") as ins:
    for line in ins:
        if title == '':
            title = line
            continue
        line = line.strip()
        features = line.split(',')
        if 'N/A' in features:
            continue
        features[0] = sex.index(features[0])
        features[1] = int(features[1]) / 10
        features[2] = race.index(features[2])
        features[3] = int(features[3])
        features[4] = int(features[4])
        features[5] = int(features[5])
        features[6] = int(features[6]) / 10
        features[7] = int(features[7]) / 100
        features[8] = int(features[8]) / 100
        features[9] = c_charge_degree.index(features[9])
        features[10] = int(features[10])
        features[11] = r_charge_degree.index(features[11])
        features[12] = int(features[12])
        features[13] = int(features[13])
        if int(features[14]) <= 5:
            features[14] = 0
        else:
            features[14] = 1
        data.append(features)
data = np.asarray(data)
print(len(data))
print(np.sum(data[:, 14]))
np.savetxt("../datasets/compas.csv", data, fmt="%d", delimiter=",")
