import io, sys, time, warnings, multiprocessing
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

data = pd.read_csv('cleaned_data.csv', low_memory=False)
feature_names = np.concatenate((np.array(['intercept']), np.array(data.columns)), axis=0)
feature_names = np.delete(feature_names, feature_names=='Birth Weight')
warnings.filterwarnings("ignore")

def experiment(input):
    # Set random seed
    np.random.seed(input)

    length_split = np.zeros(len(full_sample) + 1)
    length_randomized = np.zeros(len(full_sample) + 1)
    coverage_split = np.full(len(full_sample), np.nan)
    coverage_randomized = np.full(len(full_sample), np.nan)

    try:
        # Define the response variable and covariates
        data_sampled = data.sample(n=500)
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

        # -------------------------- splitting -------------------------
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
        selected_set_split = selected_set
        selected_size = len(selected_set)

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
        lci_split, uci_split = infere_model['normal_ci'][:, 0], infere_model['normal_ci'][:, 1]

        # ------------------------- randomized -------------------------
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
        selected_set_randomized = selected_set

        # inference
        exact_grid_inf = exact_grid_inference(query_spec, target_spec)
        lci_randomized, uci_randomized = exact_grid_inf._intervals(level=0.90)

        # ------------------------- results -------------------------
        for j, f in enumerate(full_sample):
            if f in selected_set_split:
                idx = np.where(selected_set_split == f)[0][0]
                length_split[j] = uci_split[idx] - lci_split[idx]
                coverage_split[j] = np.where(((lci_split[idx] > 0) | (uci_split[idx] < 0)), 1, 0)

            if f in selected_set_randomized:
                idx = np.where(selected_set_randomized == f)[0][0]
                length_randomized[j] = uci_randomized[idx] - lci_randomized[idx]
                coverage_randomized[j] = np.where(((lci_randomized[idx] > 0) | (uci_randomized[idx] < 0)), 1, 0)

        length_split[len(full_sample)] = np.mean(uci_split - lci_split)
        length_randomized[len(full_sample)] = np.mean(uci_randomized - lci_randomized)

        result = f"Task {input} completed"

    except Exception as e:
        result = f"Task {input} failed with error: {e}"

    return length_split, length_randomized, coverage_split, coverage_randomized, result

target_successes = 100
success_count = 0
task_count = 0

full_sample = [20, 21, 33, 39, 42, 45]
length_split_list = []
length_randomized_list = []
coverage_split_list = []
coverage_randomized_list = []


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(processes=8) as pool:
        while success_count < target_successes:
            future_result = pool.apply_async(experiment, (task_count,))
            try:
                length_split, length_randomized, coverage_split, coverage_randomized, result = future_result.get(timeout=30)
                if "completed" in result:
                    success_count += 1

                    length_split_list.append(length_split)
                    length_randomized_list.append(length_randomized)

                    coverage_split_list.append(coverage_split)
                    coverage_randomized_list.append(coverage_randomized)

                    print(f"Success count: {success_count}")
            except multiprocessing.TimeoutError:
                print(f"Task {task_count} timed out. Retrying...")
            task_count += 1

    print('------------------ summary ------------------ ')
    column_names = np.append(feature_names[full_sample], 'Average')
    length_split_list = pd.DataFrame(length_split_list, columns=column_names)
    length_randomized_list = pd.DataFrame(length_randomized_list, columns=column_names)
    print(np.mean(length_split_list))
    print(np.mean(length_randomized_list))

    coverage_split_list = pd.DataFrame(coverage_split_list, columns=feature_names[full_sample])
    coverage_randomized_list = pd.DataFrame(coverage_randomized_list, columns=feature_names[full_sample])
    print(np.nansum(coverage_split_list, axis=0) / np.sum(~np.isnan(coverage_split_list), axis=0))
    print(np.nansum(coverage_randomized_list, axis=0) / np.sum(~np.isnan(coverage_randomized_list), axis=0))

    length_split_list.to_csv('length_split_multi10.csv', index=False)
    length_randomized_list.to_csv('length_randomized_multi10.csv', index=False)
