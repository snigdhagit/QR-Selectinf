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

# Randomly select a few observations
data = pd.read_csv('cleaned_data.csv', low_memory=False)
data_sampled = data.sample(n=500)
cols_binary = data_sampled.columns[(data_sampled.nunique() == 2) & (data_sampled.isin([0, 1]).all())]
print(pd.DataFrame(np.column_stack((cols_binary,
                                    [data[col].value_counts().min() for col in cols_binary],
                                    [data_sampled[col].value_counts().min() for col in cols_binary]))))

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Define the response variable and covariates
Y = np.array(data_sampled['Birth Weight'])
X = data_sampled.drop(columns=['Birth Weight'])

# Standardize X
feature_names = np.concatenate((np.array(['intercept']), np.array(X.columns)), axis=0)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.c_[np.ones(np.shape(X)[0]), X]

# Standardize Y
Y = (Y - np.mean(Y)) / np.std(Y)

n, p = np.shape(X)
tau = .1
lambda_cont = .4
print(f'\nThe sample size is {n}.')
print(f'The demension is {p}.')

# ---------------------------- naive ---------------------------
print('------------------ naive ------------------ ')
# selection
select_h = max(0.05, np.sqrt(tau * (1 - tau)) * (np.log(p) / n) ** 0.25)
selected_fit = high_dim(X,
                        Y,
                        intercept=False).l1(h=select_h,
                                            tau=tau,
                                            kernel="Gaussian",
                                            Lambda=lambda_cont * np.sqrt(np.log(p) / n),
                                            standardize=False)
selected_set = np.nonzero(selected_fit['beta'])[0]
selected_size = len(selected_set)
print(selected_set)
# print(feature_names[selected_set])

# inference
# infere_h = ((selected_size + np.log(n)) / n) ** 0.4
infere_h = select_h
infere_model = low_dim(X[:, selected_set],
                       Y,
                       intercept=False).norm_ci(h=infere_h,
                                                tau=tau,
                                                alpha=0.1,
                                                kernel="Gaussian",
                                                standardize=False)
lci, uci = infere_model['normal_ci'][:, 0], infere_model['normal_ci'][:, 1]
# print(lci)
# print(uci)

selected_infere = np.zeros(p + 1)
selected_infere[selected_set] = (lci > 0) | (uci < 0)
selected_infere_set = np.nonzero(selected_infere)[0]
print(selected_infere_set)
print(feature_names[selected_infere_set])
print(np.mean(uci - lci))

df_naive = pd.DataFrame({'selected_set': selected_set,
                         'feature_names': feature_names[selected_set],
                         'lci': lci,
                         'uci': uci})

# -------------------------- splitting -------------------------
print('------------------ splitting ------------------ ')
# splitting
sample_proportion = 2 / 3
select_n = int(sample_proportion * n)
infere_n = n - select_n
index_select = np.random.choice(n, select_n, replace=False)
index_infere = np.array([i for i in range(n) if i not in index_select])
X_select, Y_select = X[index_select, :], Y[index_select]
X_infere, Y_infere = X[index_infere, :], Y[index_infere]

df_select, df_infere = pd.DataFrame(X_select), pd.DataFrame(X_infere)
df_select.columns, df_infere.columns = feature_names, feature_names
# print(pd.DataFrame(np.column_stack((cols_binary,
#                                     [df_select[col].value_counts().min() for col in cols_binary],
#                                     [df_infere[col].value_counts().min() for col in cols_binary]))))

# selection
select_h = max(0.05, np.sqrt(tau * (1 - tau)) * (np.log(p) / select_n) ** 0.25)
selected_fit = high_dim(X_select,
                        Y_select,
                        intercept=False).l1(h=select_h,
                                            tau=tau,
                                            kernel="Gaussian",
                                            Lambda=lambda_cont * np.sqrt(np.log(p) / select_n),
                                            standardize=False)
selected_set = np.nonzero(selected_fit['beta'])[0]
selected_size = len(selected_set)
print(selected_set)

# inference
# infere_h = ((selected_size + np.log(infere_n)) / infere_n) ** 0.4
infere_h = select_h
infere_model = low_dim(X_infere[:, selected_set],
                       Y_infere,
                       intercept=False).norm_ci(h=infere_h,
                                                tau=tau,
                                                alpha=0.1,
                                                kernel="Gaussian",
                                                standardize=False)
lci, uci = infere_model['normal_ci'][:, 0], infere_model['normal_ci'][:, 1]

selected_infere = np.zeros(p + 1)
selected_infere[selected_set] = (lci > 0) | (uci < 0)
selected_infere_set = np.nonzero(selected_infere)[0]
print(selected_infere_set)
print(feature_names[selected_infere_set])
print(np.mean(uci - lci))

df_split = pd.DataFrame({'selected_set': selected_set,
                         'feature_names': feature_names[selected_set],
                         'lci': lci,
                         'uci': uci})

# ------------------------- randomized -------------------------
print('------------------ randomized ------------------ ')
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
print(p)
print(n)
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

print('------------------ summary ------------------ ')
full_sample = [20, 21, 33, 39, 42, 45]
print(feature_names[full_sample])

df_naive_common_ = df_naive[df_naive.iloc[:, 0].isin(full_sample)]
df_split_common_ = df_split[df_split.iloc[:, 0].isin(full_sample)]
df_randomized_common_ = df_randomized[df_randomized.iloc[:, 0].isin(full_sample)]

print(np.mean(df_naive_common_['uci'] - df_naive_common_['lci']))
print(np.mean(df_split_common_['uci'] - df_split_common_['lci']))
print(np.mean(df_randomized_common_['uci'] - df_randomized_common_['lci']))

df_naive_common_.to_csv('naive_sampled.csv', index=False)
df_split_common_.to_csv('split_sampled.csv', index=False)
df_randomized_common_.to_csv('randomized_sampled.csv', index=False)
