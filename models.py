from used_packages import *

#%%
def svm_model(X_train, X_test, y_train, y_test, C_start=0.2, C_stop=200, step=20, cv=5):
    conf_matrices = []
    scores = []
    cv_scores = []

    C_regul = np.linspace(C_start, C_stop, step)

    # cross-validation splits
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=20)

    # loop over different values of the regularization parameter C
    for c in C_regul:
        np.random.seed(20)

        svm = SVC(C=c)

        # cross-validation
        cv_score = cross_val_score(svm, X_train, y_train, cv=cv, n_jobs=-1, scoring='f1_macro')
        cv_scores.append(cv_score)
        # Train on the full training set and test on the test set
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)

        # Calculate the accuracy on the test set
        score = f1_score(y_test, pred, average='macro')
        scores.append(score)
        # Print the results

        # Compute and display the confusion matrix for the test set
        conf_matrix = confusion_matrix(y_test, pred)
        conf_matrices.append(conf_matrix)
        print(f'C={np.round(c, 2)}, Cross-Validation F1-macro: {np.mean(cv_score) * 100:.2f}%, Test F1-macro: {score * 100:.2f}%')
        print(f'\tCV scores: {np.round(cv_score, 4) * 100}\n')

    idx_max = np.argmax(scores)
    score_max = scores[idx_max]
    C_max = C_regul[idx_max]
    cv_score_max = np.mean(cv_scores[idx_max])
    conf_matrix_max = conf_matrices[idx_max]

    print('\nBest Hyperparameters:')
    print(f'C={np.round(C_max, 2)}, Cross-Validation F1-macro: {cv_score_max * 100:.2f}%, Test F1-macro: {score_max * 100:.2f}%')
    print(f'Confusion Matrix:\n{conf_matrix_max}\n')

    return C_max, score_max, conf_matrix_max

#%%

def pca_svm(X_train, X_test, y_train, y_test, C_start=5, C_stop=30, step=5, pc_start=1, pc_stop=40, cv=5, gauss_sigm=7):

    mean_tr = X_train.mean(0)
    diffs_tr = X_train - mean_tr
    Cov = np.cov(diffs_tr.T)
    Cov_s = gaussian_filter(Cov, sigma=gauss_sigm)
    l, psi = np.linalg.eigh(Cov_s)  #l = eigenvalues, psi = eigenvectors

    # cross-validation splits
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=20)

    PC_set = range(pc_start, pc_stop+1)
    score_max = []
    score_cv_max = []

    C_max = []
    for iter, i in enumerate(PC_set):
        C_regul = np.linspace(C_start,C_stop,step)
        scores = []
        score_cv = []
        for j in C_regul:
            np.random.seed(20)
            PCs = psi[:, -i:]
            FPC_scores_tr = diffs_tr.dot(PCs)

            # project test data to lower dimensional space using
            # rotation matrix (eigenvectors) obtained from training data

            mean_ts = X_test.mean(0)
            diffs_ts = X_test - mean_ts
            FPC_scores_ts = diffs_ts.dot(PCs)

            # fit svm model
            svm = SVC(C=j)
            # cross-validation
            cv_s = cross_val_score(svm, FPC_scores_tr, y_train, cv=cv, n_jobs=-1, scoring='f1_macro')
            score_cv.append(cv_s)


            svm.fit(FPC_scores_tr, y_train)
            pred = svm.predict(FPC_scores_ts)
            conf = confusion_matrix(y_test, pred)
            scr = f1_score(y_test, pred, average='macro'),
            scores.append(scr)
        print(f'PC={i}, C={C_regul[np.argmax(scores)]}, Cross-Validation F1-macro = {np.mean(cv_s) * 100:.2f}%, Test F1-macro = {np.max(scores) * 100:.2f}%, ')
        print(f'\tCV scores: {np.round(cv_s, 4) * 100}\n')
        score_max.append(np.max(scores))
        C_max.append(C_regul[np.argmax(scores)])
        score_cv_max.append(cv_s)


    PCs = psi[:, -PC_set[np.argmax(score_max)]:]
    FPC_scores_tr = diffs_tr.dot(PCs)
    score_cv_opt = score_cv_max[np.argmax(score_max)]

    mean_ts = X_test.mean(0)
    diffs_ts = X_test - mean_ts
    FPC_scores_ts = diffs_ts.dot(PCs)

    SVM = SVC(C=C_max[np.argmax(score_max)])
    SVM.fit(FPC_scores_tr, y_train)
    pred = SVM.predict(FPC_scores_ts)
    conf = confusion_matrix(y_test, pred)

    PC_opt = PC_set[np.argmax(score_max)]
    score_opt = np.max(score_max)
    C_opt = C_max[np.argmax(score_opt)]

    print('\nBest Hyperparameters:')
    print(f'PCs: {PC_opt}, Cross-Validation F1-macro = {np.mean(score_cv_opt) * 100:.2f}%, F1-score = {score_opt * 100:.2f}%, C = {C_opt}')
    print(f'    CV scores: {np.round(score_cv_opt, 4) * 100}\n')
    print(f'Confusion Matrix:\n{conf}\n')

    return PC_opt, score_opt, C_opt, conf


