# Optimization-of-Gaussian-Process-Regression

This project consists of parallel implementations of the Gaussian Process Regression (GPR) technique to predict the value of a function at a point in a two-dimensional unit square using known values of the function at points on a grid laid out on the unit square. 
This project explores how to compute the hyper-parameters that can be used in the GPR model to predict values with high accuracy. The prediction 𝑓∗ at a point q(x,y) is given as 

*𝑓∗=𝑘∗(𝑡𝐼+𝐾)−1𝑓*

where K is the kernel matrix that represents correlation between the function values f at grid points. Specifically, for two points r(x,y) and s(u,v):

(𝑟,𝑠)=1√2𝜋𝑒−((𝑥−𝑢)22𝑙12+(𝑦−𝑣)22𝑙22)


 in which , l1 and l2 are two hyper-parameters that should be chosen to maximize the likelihood of the prediction being accurate. The vector f denotes the observed data values at the grid points. The vector 𝑘∗ is computed as given below:

𝑘∗(𝑞,𝑠)=1√2𝜋𝑒−((𝑥−𝑢)22𝑙12+(𝑦−𝑣)22𝑙22), 

for all grid points s. 
To estimate l1 and l2 we split the data into two sets randomly: 90% of the points form the training set and the remaining 10% form the test set. We select initial values for the parameters l1 and l2 and construct K using points in the training set. Next, we predict at each test point using Eq. (1). Using predictions at all the test points, we compute the mean square error (mse) of the predictions from the observed data:

𝑚𝑠𝑒=1𝑛𝑡Σ(𝑓∗(𝑟𝑖)−𝑓(𝑟𝑖))2𝑛𝑡𝑖=1, 

where nt is the number of test points. The goal is to determine those values of l1 and l2 that minimizes Mean Square Error. 
