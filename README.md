# Physics-informed-neural-networks-for-Options-price-prediction
The price of European Call Options is predicted by using The Black-Scholes model. This is a PDE (partial differential equation) model where the option price $V$ is a function of two variables: the time $t$ and the stock price $S$, initial and boundary conditions are also part of the model. The Black-Scholes model:

PDE: $\frac{\partial V}{\partial t}+\frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2}+rS\frac{\partial V}{\partial S}-rV=0, \quad 0 < t < t_{max}, \quad 0 < S < S_{max}$

Initial condition: $V(t=t_{max},S)=\max{\\{S-K,0}\\}$.

Boundary condition: $V(t,S=0)=0, \quad V(t,S=S_{max})=S_{max}-Ke^{-r(t_{max}-t))}$.


Physics-informed neural networks (PINN) are used to numerically solve the PDE and the solution provides the predictions. The implementation is done using TensorFlow. It is not a standard neural network (NN) implementation, where the NN model is trained (and the loss function is minimized) by using labeled data. Instaed, the PDE model is used to train the NN model. PINN involves random sampling of points from the domain of the variabels (t and S), which in this work are uniformly disterbuted (more advanced sampling stratagies are known to segnificantly affect the accuracy of the predictions for harder problems). Points are sampled from within the doamin (called the residual points) and also from its bounadries (i.e., according to the boundary and intial conditions). The NN model is evaluated on the sampling points and is then differentiated (using automatic diffrentation) according to the PDE. This allows to check how far the NN model is from setasfing the PDE (to consrtact a loss term) and to train the model.
5000 sampling points were used for the residuals and 100 sampling points for each of the boundary/intial conditions. The NN model includes 4 layers with 50 neurons each and an additional output layer. The activation function is $\tanh$ and the loss function uses the mse (mean square error) metric.  Finaly, the analytical solution of the model is generated and compared to the PINN-based numerical solution.
Two references were used for this work. The first (https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb#scrollTo=IHlpz-ZtZEkq) was used as a a general reference of using PINN to solve PDEs. The second (https://medium.com/@andeyharsha15/deep-neural-networks-for-solving-differential-equations-in-finance-da662ef0681) was used to to get the values for the parameters of the model (r, sigma, K) and to get the analytical soultion of the model.

$\sqrt{3x-1}+(1+x)^2$
