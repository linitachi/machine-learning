import numpy as np

mu, sigma = 0, 1  # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)  # Create Gaussian Noise

mu = s.sum()/1000
sigma = 0
for i in range(len(s)):
    sigma = sigma + ((s[i]-mu)**2)
sigma = sigma/1000
sigma = sigma**0.5
K = 0
for i in range(len(s)):
    K = K + (s[i]-mu)**4
K = K/1000/((sigma)**4)
print(K-3)
