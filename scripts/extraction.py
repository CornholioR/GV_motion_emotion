import pandas as pd

# df = pd.read_csv('data/MotionCaptureData trc/test.csv')

# df = df.drop(0)
# df = df.drop(1)
# df = df.drop(2)
# df = df.drop(3)
# df = df.drop(4)
# df['Label'] = 0

# print(df.head)

import os

directory = "CSV"
i = 0
data_dict = {}
for filename in os.listdir(directory):
    i += 1
    f = os.path.join(directory, filename)
    emotion = filename[-11:-8]

    df = pd.read_csv(
        "CSV/" + filename
    )
    df = df.drop(0)
    df = df.drop(1)
    df = df.drop(2)
    df = df.drop(3)
    df = df.drop(4)
    column_indices_to_keep = [j for j in range(150)]
    column_indices_to_keep = column_indices_to_keep[2:11]
    df_filtered = df.iloc[:, column_indices_to_keep]
    # df_filtered["Label"] = emotion
    data_dict[i] = df_filtered




