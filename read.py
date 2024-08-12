import scipy.io
mat = scipy.io.loadmat('./data/attrann.mat')
attrs = mat["attrann"][0][0]

print(attrs[0])
