test = pd.read_csv(cwd + '/data/raw/test.tsv', sep='\t', encoding='latin_1', header=None)
test.columns = ['hate_speech', 'off_lang', 'text']
from tqdm import tqdm

features = []
labels_test_lang = []
labels_test_hate = []

for i in tqdm(range(0, len(test))):
    features.append(nlp(test.text[i]).vector)
    labels_test_lang.append(test.off_lang[i])
    labels_test_hate.append(test.hate_speech[i])

features_test_lg = np.array(features)
labels_test_gs_lang_lg = np.array(labels_test_lang)
labels_test_gs_hate_lg = np.array(labels_test_hate)

feat_lg = features_test_lg.reshape(-1, features_test_lg.shape[1])
pred_hate_test_lg = model_hate_speech_MLP_lg.predict(feat_lg).ravel()
pred_off_test_lg = model_off_language_MLP_lg.predict(feat_lg).ravel()

print(scipy.stats.pearsonr(pred_hate_test_lg, labels_test_gs_hate_lg))

print(scipy.stats.pearsonr(pred_off_test_lg, labels_test_gs_lang_lg))
