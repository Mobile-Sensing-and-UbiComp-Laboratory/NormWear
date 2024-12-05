import time
import numpy as np
# from scipy import stats

from sklearn.multioutput import MultiOutputRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, Ridge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, precision_score, recall_score

# from sklearn.preprocessing import MinMaxScaler

def linear_prob(
    x_train,
    y_train,
    x_test,
    y_test,
    task_type='class',
    random_state=42
):  
    # init linear model
    if task_type == 'class':
        lp = LogisticRegression(
            max_iter=500,
            solver='newton-cg',
            # solver='sag',
            # penalty=None,
            # C=1e6, # very strong
            C=1e0, # so so
            # C=2e1, # very weak
            # C=20e1, # very weak
            # random_state=random_state,
            # class_weight='balanced'
        )
        # print(set(y_train))
    else:
        # lp = LinearRegression()
        lp = Ridge(
            max_iter=500,
            solver="cholesky",
            alpha=1e0
            # alpha=1e1
        )
        # lp = SGDRegressor(
        #     max_iter=1000,
        #     # penalty='elasticnet',
        #     # learning_rate='optimal',
        #     # alpha=1e-2,
        #     # tol=1e-4,
        #     # loss='huber',
        #     # epsilon=1e-6
        # )
        # lp = Lasso()
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            lp = MultiOutputRegressor(lp)
        
        # # z normalize
        # y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
        # y_train = (y_train - y_mean) / y_std
        # y_test = (y_test - y_mean) / y_std

        # # min-max normalize by y_train, so the output keep consistent
        # scaler = MinMaxScaler(feature_range=(1, 10))
        # if len(y_train.shape) < 2:
        #     y_train = np.reshape(y_train, (-1, 1))
        #     y_test = np.reshape(y_test, (-1, 1))
        # scaler.fit(y_train)
        # y_train = scaler.transform(y_train)
        # y_test = scaler.transform(y_test)

        # log scale
        # print(np.isnan(np.log(y_train+1)).sum() / len(y_train))
        # print(np.isnan(np.log(y_train+1)).sum(), len(y_train))
        # print(np.isnan(np.log(y_test+1)).sum() / len(y_test))
        # print(np.isnan(np.log(y_test+1)).sum(), len(y_test))
        # exit()

        y_train = np.nan_to_num(np.log(y_train+1))
        y_test = np.nan_to_num(np.log(y_test+1))
    
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    # # shuffle
    # indices = np.arange(x_train.shape[0])
    # np.random.shuffle(indices)
    # x_train = x_train[indices]
    # y_train = y_train[indices]
    
    # fit linear model
    # start = time.time()
    # print("Fitting Linear Model...")
    lp = lp.fit(x_train, y_train)
    # end = time.time()
    # print("Time consumed:", end-start, "s")

    # test time
    return calculate_score(lp, x_test, y_test, task_type, y_train=y_train)

def calculate_score(lp, x_test, y_true, task_type, y_train=None):
    if task_type == "reg":
        y_pred = lp.predict(x_test)

        # filterout nan
        nan_to_mean = np.isnan(y_pred) * np.mean(y_train, axis=0)
        y_pred = np.nan_to_num(y_pred) + nan_to_mean

        # ### simple-mean ###
        # y_mean = np.mean(y_train, axis=0)
        # y_pred = np.array([y_mean for _ in range(len(y_true))])

        # numerical stability
        y_pred = np.clip(y_pred, np.min(y_train), np.max(y_train))
        # if len(y_pred.shape) > 1 and y_pred.shape[1] < 2:
        #     y_pred = y_pred[:, 0]

        # print(np.mean(y_pred), np.min(y_pred), np.max(y_pred))

        final_scores = [1 - np.mean(np.absolute((y_true - y_pred) / y_true))]
        return final_scores
    else:
        y_pred = lp.predict_proba(x_test)

        final_scores = list()

        # roc auc
        y_set = list(set(y_true))
        y_pred_class = np.argmax(y_pred, axis=1)

        # ### simple-mode ###
        # mode = stats.mode(y_train).mode
        # y_pred_class = np.array([mode for _ in range(len(y_true))])
        # one_hot = np.zeros(y_pred.shape)
        # one_hot[np.arange(len(y_pred)), y_pred_class] = 1
        # y_pred = one_hot

        # remove non-exist class in y_true
        y_pred = y_pred[:, y_set]
        y_pred /= np.sum(y_pred, axis=1, keepdims=True)

        # calculate score
        if len(y_set) <= 2:
            final_scores.append(roc_auc_score(y_true, y_pred[:, 1]))
            final_scores.append(average_precision_score(y_true, y_pred[:, 1]))
        else:
            final_scores.append(roc_auc_score(y_true, y_pred, multi_class="ovo", average="macro", labels=y_set))
            final_scores.append(average_precision_score(y_true, y_pred, average="macro"))
        
        final_scores.append(np.mean((y_true == y_pred_class)))
        final_scores.append(precision_score(y_true, y_pred_class, average='macro'))
        final_scores.append(recall_score(y_true, y_pred_class, average='macro'))
        final_scores.append(f1_score(y_true, y_pred_class, average="macro"))
        
        # final_scores.append(balanced_accuracy_score(y_true, y_pred_class))
        
        return final_scores