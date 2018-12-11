import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import random
import h2o
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning) # arises with current lightGBM + numpy
random.seed(11)

def load_apps(application_filename, is_train_set):
    apps_df = pd.read_csv(application_filename)
    apps_df.columns = [str.lower(column) for column in apps_df.columns]
    # convert Y/N to True/False booleans:
    apps_df['flag_own_car'] = apps_df.flag_own_car=='Y'
    apps_df['flag_own_realty'] = apps_df.flag_own_realty=='Y'
    # convert gender M/F flag to boolean:
    apps_df['gender_female'] = apps_df['code_gender']=='F'
    apps_df.drop('code_gender', axis=1, inplace=True)
    # convert mistyped float/str column:
    apps_df['ext_source_1'] = apps_df.ext_source_1.astype(float)

    # we're going to rename housing age and move it to the end so that it's kept separate from our housing quality score
    apps_df['housing_age'] = apps_df.years_build_medi
    apps_df.drop('years_build_medi', axis=1, inplace=True)

    # the nulls for who was with the person when they applied should be 'Unaccompanied'
    apps_df['name_type_suite'].fillna(value='Unaccompanied', inplace=True)

    housing_cols = apps_df.columns[43:84]
    housing_categoricals = ['fondkapremont_mode', 'housetype_mode', 'wallsmaterial_mode', 'emergencystate_mode']
    apps_df['housing_null_count'] = apps_df[housing_cols].isnull().sum(axis=1)
    apps_df['other_null_count'] = apps_df.isnull().sum(axis=1)
    apps_df['housing_quality'] = apps_df[housing_cols].sum(axis=1)
    # NOTE: the above sum function effectively imputes 0s for nulls, which is okay
    # since no basement area generally means no basement

    if is_train_set:
        apps_df = apps_df[apps_df.amt_income_total < 1e7]

    return apps_df

def add_eng_cols(df, log=True):
    df['payment_rate'] = df.amt_annuity / df.amt_credit
    df['relative_annuity'] = df.amt_annuity / (df.amt_income_total + 1)  # +1 to avoid dividing by 0
    df['relative_payment_rate'] = df.payment_rate / (df.amt_income_total + 1)
    df['pct_days_employed'] = df.days_employed / (df.days_birth-21.1*365) # % of adulthood employed
    df['housing_per_income'] = df.housing_quality / df.amt_income_total
    df['ext_mean'] = df[['ext_source_1','ext_source_2','ext_source_3']].mean(axis=1)
    df['ext_std'] = df[['ext_source_1','ext_source_2','ext_source_3']].std(axis=1)
    df['income_per_dependent'] = df.amt_income_total / (1 + df.cnt_children)

    if log:
        # log transformations from the "examining the data" section
        df['amt_credit_log'] = df['amt_credit'].apply(np.log10)
        df['amt_income_total_log'] = df['amt_income_total'].apply(np.log10)
        # not actually very useful in empirical tests - but we'll weed out unimportant features later

    return df

def load_previous(previous_filename):
    # LOAD AND CLEAN:
    previous_df = pd.read_csv(previous_filename)
    previous_df.columns = [str.lower(column) for column in previous_df.columns]

    # fill accompanying person and downpayment appropriately
    previous_df['name_type_suite'].fillna(value='Unaccompanied', inplace=True)

    # convert Y/N to boolean
    previous_df['flag_last_appl_per_contract'] = previous_df['flag_last_appl_per_contract']=='Y'

    # drop three columns that are overwhelmingly null
    previous_df.drop(['rate_down_payment',
                      'rate_interest_primary',
                      'rate_interest_privileged',
                      'name_cash_loan_purpose'],
                     axis=1, inplace=True)
    return previous_df

def load_pcb(pcb_filename):
    #load
    pcb_df = pd.read_csv(pcb_filename)
    pcb_df.columns = [str.lower(column) for column in pcb_df.columns]

    # some renaming for clarity:
    pcb_df.rename(columns={
        'sk_dpd':'av_dpd_minor',
        'sk_dpd_def':'av_dpd_major'
        }, inplace=True)

    # rename name_contract_status to avoid confusion with previous_df column of same name:
    # (not doing this creates an evil combinatorial-column-proliferation problem when groupby-ing the combined table)
    pcb_df.rename(columns={'name_contract_status':'pcb_name_contract_status'}, inplace=True)

    # compress pcb_df to one row per loan, compiling mean and var for days late
    pcb_df = pcb_df[['sk_id_prev', 'av_dpd_minor', 'av_dpd_major','months_balance', 'pcb_name_contract_status']]
    pcb_df = pcb_df[~pcb_df.pcb_name_contract_status.isin(['Canceled', 'XNA'])] # these 17 rows aren't worth 2 new dummy cols
    pcb_df = pd.get_dummies(pcb_df, 'pcb_name_contract_status')
    pcb_df = pcb_df.groupby('sk_id_prev', as_index=False).agg(['mean', 'max', 'var'])
    # flatten multiindex
    pcb_df.columns = ['_'.join(col) for col in pcb_df.columns]
    pcb_df.reset_index(inplace=True)
    return pcb_df

def load_install_payments(install_payments_filename):
    # load
    install_payments_df = pd.read_csv(install_payments_filename)
    install_payments_df.columns = [str.lower(column) for column in install_payments_df.columns]

    # create combination columns
    install_payments_df['amt_unpaid'] = install_payments_df.amt_instalment - install_payments_df.amt_payment
    install_payments_df['pct_unpaid'] = (install_payments_df.amt_instalment - install_payments_df.amt_payment) \
                                    / install_payments_df.amt_instalment
    install_payments_df['days_late'] = install_payments_df.days_entry_payment - install_payments_df.days_instalment

    # drop columns we've incorporated into combination columns or otherwise don't have use for
    install_payments_df.drop(['num_instalment_version',
                              'amt_payment',
                              'days_entry_payment',
                              'sk_id_curr'], axis=1, inplace=True)

    # group by loan
    agg_funcs = {'num_instalment_number':'count',
                'days_instalment':['min', 'mean'],
                'pct_unpaid':['mean', 'max'],
                 'amt_unpaid':['mean', 'var', 'max'],
                'days_late':['mean', 'max']}
    install_payments_df = install_payments_df.groupby('sk_id_prev', as_index=False).agg(agg_funcs)
    # flatten multiindex
    install_payments_df.columns = ['_'.join(col) for col in install_payments_df.columns]
    install_payments_df.rename(columns={'sk_id_prev_':'sk_id_prev'}, inplace=True)
    return install_payments_df

def load_cc_bal(cc_bal_filename):
    # load
    cc_bal_df = pd.read_csv(cc_bal_filename)
    cc_bal_df.columns = [str.lower(column) for column in cc_bal_df.columns]

    # generate pcdt_unpaid column
    cc_bal_df['pct_unpaid'] = (cc_bal_df.amt_balance - cc_bal_df.amt_payment_current) / (cc_bal_df.amt_balance + 1)

    # drop sk_id_curr (we'll groupby and later join with sk_id_prev)
    # cc_bal_df.drop('sk_id_curr', axis=1, inplace=True)

    # these 22 rows out of 3 mil aren't worth 2 new dummy cols:
    cc_bal_df = cc_bal_df[~cc_bal_df.name_contract_status.isin(['Refused', 'Approved'])]
    cc_bal_df = pd.get_dummies(cc_bal_df, 'name_contract_status')

    # because this dataset is pretty opaque, we're just going to take the mean and var of every column for each user
    cc_bal_df = cc_bal_df.groupby('sk_id_prev', as_index=False).agg(['mean','var'])
    cc_bal_df.columns = ['_'.join(col) for col in cc_bal_df.columns]
    # but we don't need variance for the binary dummy columns, or for sk_id_curr:
    cc_bal_df.drop(['name_contract_status_Active_var',
                    'name_contract_status_Completed_var',
                    'name_contract_status_Demand_var',
                    'name_contract_status_Sent proposal_var',
                    'name_contract_status_Signed_var',
                    'sk_id_curr_var'], axis=1, inplace=True)
    cc_bal_df.rename(columns={'sk_id_curr_mean':'sk_id_curr'}, inplace=True)
    cc_bal_df.reset_index(inplace=True) # because for some reason pandas ignores the as_index=False option above

    # cc_bal_df.sk_id_prev.nunique() is 104,307
    # cc_bal_df.sk_id_curr.nunique() is 103,558
    # therefore grouping by sk_id_curr will only collapse less than 1% of the rows.  We'll just take a simple mean.
    cc_bal_df = cc_bal_df.groupby('sk_id_curr', as_index=False).agg('mean')
    return cc_bal_df

# define utility functions for splitting prev (most recent) vs past and grouping past applications by applicant
def split_past_prev(df, time_col, past_id, prev_suffix=None):
    '''
    time_col = column that contains days since application
    past_id = column that contains id of the loan represented by the row
    '''
    start = time()
    df = df.sort_values(by=['sk_id_curr', time_col]) # each group sorted by days_decision so most recent is last
    prev_df = df.groupby('sk_id_curr').nth(-1) # selects last entry from each group
    past_df = df[~df[past_id].isin(prev_df[past_id])] # selects all entries not in previous_df
    prev_df.reset_index(inplace=True)
    if prev_suffix:
        prev_df.columns = [col + prev_suffix for col in prev_df.columns]
    prev_df.rename(columns={('sk_id_curr'+prev_suffix):'sk_id_curr'}, inplace=True)
    print("Elapsed time (minutes):", round((time()-start)/60, 1))
    return (past_df, prev_df)

def categorical_mode(series):
    if series.count() == 0:
        return float('NaN')
    else:
        return series.mode()[0]

def group_apps(df, count_col, cats_to_encode=[], summary_suffix=None):
    start = time()

    # eliminate spaces from column names:
    df.columns = [col.replace(' ', '_') for col in df.columns]

    cat_cols = []
    for col in df.columns:
        if df.dtypes[col] == 'object' and col not in cats_to_encode:
            cat_cols.append(col)
    num_cols = [col for col in df.columns if col not in (cat_cols + cats_to_encode + ['sk_id_curr'])]

    df['target'] = y # NOTE: sloppy addition - if re-using, note that y should be passed in
    for col in cats_to_encode:
        df[col + '_target_mean'] = target_mean_encode(col, 'target', df)
        df[col + '_target_std'] = target_std_encode(col, 'target', df)

    df = pd.get_dummies(df, columns=cats_to_encode)
    # eliminate spaces from column names:
    df.columns = [col.replace(' ', '_') for col in df.columns]
    dummy_cols = [col for col in df.columns if col not in (
        num_cols + cat_cols + cats_to_encode + ['target', 'sk_id_curr'])]

    agg_funcs = {}
    for col in dummy_cols:
        agg_funcs[col] = ['mean', 'var', 'sum']
    for col in num_cols:
        agg_funcs[col] = ['min', 'max', 'mean', 'var']  # alternative: ['mean', 'var']
    for col in cat_cols:
        agg_funcs[col] = [categorical_mode, 'nunique']
        # categorical_mode = custom function defined above; target_mean_encode defined higher up
    agg_funcs[count_col] = 'count'  # we'll rename this prev_app_count below

    print("Performing groupby...")
    grouped_df = df.groupby('sk_id_curr').agg(agg_funcs)  # NOTE: takes a while
    grouped_df.rename(columns={count_col:'prev_app_count'}, inplace=True)
    grouped_df.reset_index(inplace=True)

    # collapse multi-index
    grouped_df.columns = ['_'.join(col) for col in grouped_df.columns]

    # add suffix to all columns (to avoid duplicates later)
    if summary_suffix:
        grouped_df.columns = [col + summary_suffix for col in grouped_df.columns]
        # remove suffix from sk_id_curr - we need it for merging
        grouped_df.rename(columns={('sk_id_curr'+summary_suffix):'sk_id_curr'}, inplace=True)

    print("Elapsed time (minutes):", round((time()-start)/60, 1))
    return grouped_df

def load_bureau(bureau_filename):
    bureau_df = pd.read_csv(bureau_filename)
    bureau_df.columns = [str.lower(column) for column in bureau_df.columns]
    return bureau_df

def load_bureau_balance(bb_filename):
    # load
    bureau_balance_df = pd.read_csv(bb_filename)
    bureau_balance_df.columns = [str.lower(column) for column in bureau_balance_df.columns]
    # NOTE: there are only two columns that are over half null, and they're both around 70% and important enough to keep

    # drop not-so-useful column
    bureau_balance_df.drop('months_balance', axis=1, inplace=True)
    # generate dummies for the main categorical var for groupby-ing down to one entry per loan
    bureau_balance_df = pd.get_dummies(bureau_balance_df, 'status')
    # groupby and generate a total months column:
    bureau_balance_df = bureau_balance_df.groupby('sk_id_bureau', as_index=False).agg(['mean','sum'])
    bureau_balance_df['total_months'] = bureau_balance_df[bureau_balance_df.columns[1:]].sum(axis=1)
    bureau_balance_df.columns = ['_'.join(col) for col in bureau_balance_df.columns]
    bureau_balance_df.reset_index(inplace=True)
    return bureau_balance_df

def load_glrm(is_train_set):
    if is_train_set:
        return pd.read_csv('glrm_imp30_v2.csv', index_col=0)
    else:
        return pd.read_csv('glrm_imp30_v2_test.csv', index_col=0)

def load_ancillaries(load_from_files=True):

    if load_from_files:
        prev1_df = pd.read_csv('prev1_df.csv', index_col=0)
        prev2_df = pd.read_csv('prev2_df.csv', index_col=0)
        past_summary_df = pd.read_csv('past_summary_df.csv', index_col=0)
        prev_bureau1_df = pd.read_csv('prev_bureau1_df.csv', index_col=0))
        prev_bureau2_df = pd.read_csv('prev_bureau2_df.csv', index_col=0))
        bureau_summary_df = pd.read_csv('bureau_summary_df.csv', index_col=0))
        cc_bal_df = pd.read_csv('cc_bal_df.csv', index_col=0)
        return prev1_df, prev2_df, past_summary_df, prev_bureau1_df, prev_bureau2_df \
            bureau_summary_df, cc_bal_df

    # load general ancillary tables
    previous_df = load_previous("previous_application.csv")
    pcb_df = load_pcb("POS_CASH_balance.csv")
    install_payments_df = load_install_payments('installments_payments.csv')
    cc_bal_df = load_cc_bal('credit_card_balance.csv')
    all_previous_df = previous_df.merge(
        pcb_df, on='sk_id_prev', how='left').merge(
        install_payments_df, on='sk_id_prev', how='left')

    # perform table most-recent splits and overall average groupbys
    # pull prev_1 from past; save restOfPast
    other_past1_df, prev1_df = split_past_prev(all_previous_df,
                                       time_col='days_decision',
                                       past_id='sk_id_prev',
                                       prev_suffix='_prev1')
    # pull prev_2 from restOfPast
    other_past2_df, prev2_df = split_past_prev(other_past1_df,
                                       time_col='days_decision',
                                       past_id='sk_id_prev',
                                       prev_suffix='_prev2')
    # do maximalist summary of all data (including first two)
    past_summary_df = group_apps(all_previous_df,
                         count_col='sk_id_prev',
                         cats_to_encode=['name_contract_type','name_contract_status','name_payment_type'],
                         summary_suffix='_prev_summ')
    # cleanup
    del previous_df, all_previous_df, pcb_df, install_payments_df


    # load bureau ancillary tables
    bureau_df = load_bureau('bureau.csv')
    bureau_balance_df = load_bureau_balance('bureau_balance.csv')
    all_bureau_df = pd.merge(bureau_df, bureau_balance_df, on='sk_id_bureau', how='left')

    # perform table most-recent splits and overall average groupbys
    # pull prev_1 from past; save restOfPast
    other_past_bureau1_df, prev_bureau1_df = split_past_prev(all_bureau_df,
                                                           time_col='days_credit',
                                                           past_id='sk_id_bureau',
                                                           prev_suffix='_bureau_p1')
    # pull prev_2 from restOfPast
    other_past_bureau2_df, prev_bureau2_df = split_past_prev(other_past_bureau1_df,
                                                           time_col='days_credit',
                                                           past_id='sk_id_bureau',
                                                           prev_suffix='_bureau_p2')
    # do maximalist summary of all data (including first two)
    bureau_summary_df = group_apps(bureau_df,
                                 count_col='sk_id_bureau',
                                 cats_to_encode=['credit_active'],
                                 summary_suffix='_bureau_summ')
    # cleanup
    del bureau_df, bureau_balance_df, all_bureau_df, other_past_bureau1_df, other_past_bureau2_df

    return prev1_df, prev2_df, past_summary_df, prev_bureau1_df, prev_bureau2_df \
        bureau_summary_df, cc_bal_df

def get_important_cols(n=48, full_dataset_df=full_dataset_df):

    def target_seq_encode(colname, target, df):
        '''returns column with each value replaced by the integer index of the value
        in a list of values sorted by the mean of the target value for that column'''
        value_means = []
        for value in df[colname].unique():
            value_means.append( (value, df[df[colname]==value][target].mean()) )
        value_means = sorted(value_means, key=lambda x: x[1])
        # create dictionary of {value, index-in-list-sorted-by-mean}
        value_seq = {}
        i = 0
        for tup in value_means:
            value_seq[tup[0]] = i
            i+=1
        return df[colname].apply(lambda x: value_seq[x])

    X_num_encoded = full_dataset_df.copy()
    X_cat_indices = []
    for colname in X_num_encoded:
        if X_num_encoded.dtypes[colname] in ['object', 'bool']:
            X_cat_indices.append(list(X_num_encoded.columns).index(colname))
            X_num_encoded[colname] = target_seq_encode(colname, 'target', X_num_encoded)

    y = X_num_encoded['target']
    X_num_encoded.drop('target', axis=1, inplace=True) # shifts the indices above, hence the ind-1 offset in next line
    X_cat_indices = [ind-1 for ind in X_cat_indices]

    # LightGBM can process categorical cols IF they have integer values AND are explicitly declared in parameters
    import lightgbm as lgb

    # create LightGBM-specific datasets with the categorical cols explicitly declared
    lgbm_train = lgb.Dataset(X_num_encoded, label=y, categorical_feature=X_cat_indices)

    best_params = {
        'class_weight':'balanced',
        'reg_lambda': 1,
        'reg_alpha': 2,
        'random_state': 11,
        'num_leaves': 14,
        'n_estimators': 300,
        'min_samples_split': 2,
        'min_data_in_leaf': 1200,
        'boosting_type': 'goss',
        'importance_type': 'gain'
    }

    # then test LightGBM on this dataset
    start = time()
    lgbmc = lgb.train(params=best_params, train_set=lgbm_train)
    y_pred = lgbmc.predict(X_num_encoded)

    # define top features:
    importances = [round(i) for i in lgbmc.feature_importance(importance_type='gain')]
    labeled_fi = sorted( list(zip(X_num_encoded.columns, importances)), key=lambda x: -x[1])
    top_features = [tup[0] for tup in labeled_fi[:n]]
    return top_features

def glrm_imp(full_dataset_train_df, full_dataset_test_df, k, n_top_features):

    top_48_features = get_important_cols(n=n_top_features, full_dataset_df=full_dataset_train_df)

    # create column types
    type_conversions = {
        np.dtype('float64'):'real',
        np.dtype('int64'):'int',
        np.dtype('bool'):'enum',
        np.dtype('object'):'enum',
    }

    glrm_column_types = {}
    coltypes = full_dataset_train_df[top_48_features].dtypes
    for col in top_48_features:
        glrm_column_types[col] = type_conversions[coltypes[col]]
    # convert integer dummies to enum:
    for col in top_48_features:
        if full_dataset_train_df[col].dtype == np.dtype('int64') and \
        len(full_dataset_train_df[col].value_counts()) == 2:
            glrm_column_types[col] == 'enum'

    full_dataset_df = pd.concat([full_dataset_train_df, full_dataset_test_df], axis=0)

    import h2o
    from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
    h2o.init()

    # create GLRM dataset
    imp_h2 = h2o.H2OFrame(full_dataset_df[top_48_features], column_types=glrm_column_types)

    # create GLRM model
    glrmodel_imp30 = H2OGeneralizedLowRankEstimator(
                                       k=k,
                                       seed=11,
                                       transform="STANDARDIZE",
                                       regularization_x="None",
                                       regularization_y="None",
                                       init='SVD',
                                       max_iterations=30,
                                       multi_loss='categorical',
                                       #loss_by_col=[loss_by_col[c] for c in khous_run.columns],
                                       impute_original=True)

    # run and get results
    start = time()
    glrmodel_imp30.train(training_frame=imp_h2)
    model = glrmodel_imp30._model_json
    model_out = model['output']
    # num_archs = glrmodel_5.params['k']['actual']
    archs_df = model_out["archetypes"].as_data_frame()
    archs = np.array(archs_df)
    X = h2o.get_frame(model_out['representation_name'])
    glrm_df = X.as_data_frame()
    glrm_train_df = glrm_df[:-full_dataset_test.shape[0]]
    glrm_test_df = glrm_df[-full_dataset_test.shape[0]:]
    glrm_train_df.to_csv('glrm_imp30_v2_train')
    glrm_test_df.to_csv('glrm_imp30_v2_test')

    return glrm_train_df, glrm_test_df

def pare_features(X_set, y=y, importance_threshold=0.5, models=None, return_importances=False):
    '''Returns version of X_set with only features with greater av importance than
    the user-secified importance_threshold'''

    # default params with extra trees
    basic_params = {
        'class_weight':'balanced',
        'n_estimators': 400
    }

    # params produced by a basic hyperparameter sweep later in this notebook
    best_params = {
        'class_weight':'balanced',
        'reg_lambda': 1,
        'reg_alpha': 2,
        'random_state': 11,
        'num_leaves': 14,
        'n_estimators': 300,
        'min_samples_split': 2,
        'min_data_in_leaf': 1200,
        'boosting_type': 'goss'
    }

    # params taken almost verbatim from a public kernel (explained later in this notebook)
    kernel_params_reduced = {
        'class_weight':'balanced',
        'n_estimators':500,  # reduced in the interest of processing time
        'learning_rate':0.02,
        'num_leaves':15,
        'colsample_bytree':0.95,
        'subsample':0.85,
        'max_depth':8,
        'reg_alpha':0.04,
        'reg_lambda':0.073,
        'min_split_gain':0.022,
        'min_child_weight':60,
        'metric':'auc'
    }

    start = time()

    X_train, X_test, y_train, y_test = train_test_split(X_set, y, test_size=1/3)

    if not models:
        lgbmc1 = LGBMClassifier(**basic_params)
        lgbmc2 = LGBMClassifier(**best_params)
        lgbmc3 = LGBMClassifier(**kernel_params_reduced)
        models = [lgbmc1, lgbmc2, lgbmc3]

    all_importances = []

    for model in models:
        # fit model
        model.fit(X_train, y_train)
        # score and print to confirm everything works:
        y_test_proba = model.predict_proba(X_test)[:,1]
        test_auroc = round(roc_auc_score(y_test, y_test_proba), 3)
        print("  Test AUROC {}: {}".format(models.index(model), test_auroc))
        # record model importances
        all_importances.append(model.feature_importances_)

    # compile averages and important columns:
    av_importances = np.mean(all_importances, axis=0)
    labeled_av_importances = list(zip(X_set.columns, av_importances))
    important_columns = [tup[0] for tup in labeled_av_importances if tup[1] >= importance_threshold]

    if return_importances:
        return X_set[important_columns], labeled_av_importances
    else:
        return X_set[important_columns]

def get_pared_df(full_dataset_df):
    best_params = {
        'class_weight':'balanced',
        'reg_lambda': 1,
        'reg_alpha': 2,
        'random_state': 11,
        'num_leaves': 14,
        'n_estimators': 300,
        'min_samples_split': 2,
        'min_data_in_leaf': 1200,
        'boosting_type': 'goss'
    }
    # execute importance-based feature paring
    full_dummy_df = pd.get_dummies(full_dataset_df.drop(['target', 'sk_id_curr'], axis=1), dummy_na=True)
    # remove any lingering sk_id columns, which may contribute to overfitting
    full_dummy_df.drop([col for col in full_dummy_df.columns if 'sk_id' in col], axis=1, inplace=True)
    pared_df = pare_features(full_dummy_df, y, importance_threshold=0.5, return_importances=False)
    # train model on only the useless features:
    dropped_cols = [col for col in full_dummy_df if col not in pared_df]
    lgbmc2 = LGBMClassifier(**best_params)
    lgbmc2.fit(full_dummy_df[dropped_cols], full_dummy_df.target)
    labeled_fi = sorted( list(zip(dropped_cols, lgbmc2.feature_importances_)), key=lambda x: -x[1])
    pd.options.mode.chained_assignment = None  # default='warn'
    for col in [tup[0] for tup in labeled_fi[:8]]:
        pared_df[col] = full_dummy_df[col].copy()

    return pared_df

def get_boost_df(glrm_df, y):

    gnb1 = GaussianNB(priors=[0.7, 0.3])
    gnb1.fit(glrm_df, y)
    gnb2 = GaussianNB(priors=[0.35, 0.65])
    gnb2.fit(glrm_df, y)

    gnb1_pred = pd.Series(gnb1.predict(glrm_all))
    gnb1_prob = pd.Series(gnb1.predict_proba(glrm_all)[:,0])
    gnb2_pred = pd.Series(gnb2.predict(glrm_all))
    gnb2_prob = pd.Series(gnb2.predict_proba(glrm_all)[:,0])
    gnb_boost_df = pd.concat([gnb1_pred, gnb1_prob, gnb2_pred, gnb2_prob], axis=1)
    gnb_boost_df.columns = ['gnb1_flag', 'gnb1_prob', 'gnb2_flag', 'gnb2_prob']

    return gnb_boost_df

def compile_full_dataset(is_train_set, ancillaries_from_memory=True, glrm_from_memory=True):
    # load main application table
    if is_train_set:
        apps_df = load_apps('applications_train.csv', is_train_set=True)
    else:
        apps_df = load_apps('applications_test.csv', is_train_set=False)
    apps_df = add_eng_cols(apps_df, log=True)

    # load ancillary tables
    prev1_df, prev2_df, past_summary_df, prev_bureau1_df, prev_bureau2_df, bureau_summary_df, \
    cc_bal_df = load_ancillaries(load_from_memory=ancillaries_from_memory)

    # merge all tables
    full_dataset_df = apps_df.merge(
        prev1_df, on="sk_id_curr", how="left").merge(
        prev2_df, on="sk_id_curr", how="left").merge(
        past_summary_df, on="sk_id_curr", how="left").merge(
        prev_bureau1_df.reset_index(), on="sk_id_curr", how="left").merge(
        prev_bureau2_df.reset_index(), on="sk_id_curr", how="left").merge(
        bureau_summary_df.reset_index(), on="sk_id_curr", how="left").merge(
        cc_bal_df.reset_index(), on="sk_id_curr", how="left")

    # get glrm
    if glrm_from_memory:
        glrm_df = load_glrm(is_train_set=is_train_set)
    else:
        glrm_df = glrm_imp(full_dataset_df, k=30, n_top_features=50)

    # add boosting columns and return
    pared_df = get_pared_df(full_dataset_df)
    boost_additions_df = get_boost_df(glrm_df, full_dataset_df.target)
    return( pd.concat([pared_df, boost_additions_df], axis=1)) )

def get_preds(X_train, X_test, y_train, sk_id_curr_test):
    params = {
        'class_weight':'balanced',
        'n_estimators':5000,
        'learning_rate':0.015,
        'num_leaves':15,
        'colsample_bytree':0.95,
        'subsample':0.85,
        'max_depth':8,
        'reg_alpha':0.04,
        'reg_lambda':0.073,
        'min_split_gain':0.022,
        'min_child_weight':60,
        'metric':'auc'
    }
    lgbmc = LGBMClassifier(**params)
    lgbmc.fit(X_train, y_train)
    y_test_proba = lgbmc.predict_proba(X_test)[:,1]
    return pd.concat([sk_id_curr_test, y_test_proba], axis=1)


'''
CRIRICAL NOTE: must use SAME EXACT features for train and test - that means
passing in important columns, not just generating them

TO TRY:
try boosting with glrm_imp30, glrm_imp20 vs glrm_all to see which works better
try with and without dropping sk_id_curr
try adding my new categorical GNB as a booster
try increasing pared_features threshold to 3 (and then adding back the top 15-20 features from FI)
try reducing learning rate a smidgen more (0.012), and maybe increasing n_estimators to 7K
'''
