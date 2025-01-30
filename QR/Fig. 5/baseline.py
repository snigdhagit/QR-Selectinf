import io, sys, time
import numpy as np
import pandas as pd
import numpy.random as rgt
from pathlib import Path
from itertools import combinations
from conquer.linear_model import low_dim, high_dim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from selectinf.base import selected_targets
from selectinf.QR_lasso import QR_lasso
from selectinf.randomization import randomization
from selectinf.approx_reference import approximate_grid_inference
from selectinf.exact_reference import exact_grid_inference
from sklearn.preprocessing import StandardScaler

# Set random seed
np.random.seed(321)

# Define the response variable and covariates
data = pd.read_csv('cleaned_data.csv', low_memory=False)
Y = np.array(data['Birth Weight'])
X = data.drop(columns=['Birth Weight'])

# Standardize X
feature_names = np.concatenate((np.array(['intercept']), np.array(X.columns)), axis=0)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.c_[np.ones(np.shape(X)[0]), X]

# Standardize Y
Y = (Y - np.mean(Y)) / np.std(Y)

n, p = np.shape(X)
tau = .1
lambda_cont = .4 * np.sqrt(n) / 24
print(f'\nThe sample size is {n}.')
print(f'The demension is {p}.')

# selection
randomizer = randomization.isotropic_gaussian(shape=(p,),
                                              scale=(.5 / np.sqrt(n)))
conv = QR_lasso(X,
                Y,
                tau=tau,
                randomizer=randomizer,
                Lambda=lambda_cont * np.sqrt(np.log(p) / n))
conv.fit()
conv.setup_inference()
query_spec = conv.specification
target_spec, _ = selected_targets(X,
                                  Y,
                                  tau=tau,
                                  solution=conv.observed_soln)

# nonzero set of penalized estimator
selected_set = np.nonzero(conv.observed_soln)[0]
print(f'the lambda is {lambda_cont * np.sqrt(np.log(p) / n)}.')
print(selected_set)

# inference
exact_grid_inf = exact_grid_inference(query_spec, target_spec)
lci, uci = exact_grid_inf._intervals(level=0.90)

selected_infere = np.zeros(p + 1)
selected_infere[selected_set] = (lci > 0) | (uci < 0)
selected_infere_set = np.nonzero(selected_infere)[0]
print(selected_infere_set)
print(feature_names[selected_infere_set])
print(np.mean(uci - lci))

df_randomized = pd.DataFrame({'selected_set': selected_set,
                   'feature_names': feature_names[selected_set],
                   'lci': lci,
                   'uci': uci})
df_randomized.to_csv('randomized_full.csv', index=False)