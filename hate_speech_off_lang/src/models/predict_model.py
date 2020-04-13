def prediction_correlation(features_test_lg, labels_test_gs_hate_lg, model_hate_speech_MLP_lg):
    import scipy
    from scipy import stats
    feat_lg = features_test_lg.reshape(-1, features_test_lg.shape[1])
    pred_hate_test_lg = model_hate_speech_MLP_lg.predict(feat_lg).ravel()
    correl_hate = scipy.stats.pearsonr(pred_hate_test_lg, labels_test_gs_hate_lg)
    return correl_hate
