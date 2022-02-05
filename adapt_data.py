import numpy as np
file = r'/home/eric/Documents/UT/HACKATHON/data/SUBJECTS/S001/1st trial closed eyes.txt'
data = np.genfromtxt(file, delimiter = ',').transpose()
fs = 256
time = 60
print(data.shape)
data = data[:4, :min(fs*time, data.shape[1])] #We collect the first minute of data (or all data is sample is shorter)
