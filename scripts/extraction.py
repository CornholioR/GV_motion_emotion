import pandas as pd
import numpy as np

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
# data_dict = {}
dataset=[]
emotionData=[]
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
    # print(column_indices_to_keep)
    
    emotionData.append(emotion)
    df_filtered = df.iloc[:, column_indices_to_keep]
    dataset.append(df_filtered.values.tolist()[0:104])
    # print(len(df_filtered.values.tolist()[0:104]),len(df_filtered.values.tolist()[0]))
    # df_filtered["Label"] = emotion
    # data_dict[i] = df_filtered
# print(dataset)
dataset=np.array(dataset)
dataset=dataset.astype(float)
# print(dataset)
# print(emotionData)
# print(len(dataset), len(dataset[0]), len(dataset[0][0]))
# for i in range (81):
#     print(len(dataset[i]))

# print(type(dataset[0][0][0]))
