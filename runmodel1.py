import model1 
import pickle

data = pickle.load(open('trainingData.pkl', 'rb'))
F = 100 
T = 1000
alpha = 0.50
beta = 0.50
burnIn = 50
filename = 'sample_m1_F'+str(F)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'burnIn'+str(burnIn)+'.pkl'
print("Running Gibbs for", F, "frames", "with alpha=", alpha, "beta=", beta)
result = model1.gibbs(F, alpha, beta, T, burnIn, data)
pickle.dump(result, open(filename, 'wb'))
