from utils import *
from tabular_data_analysis import *
from text_retrieval import *

# Basic Stat
X = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
print('Mean: ', compute_mean(X))

X = [1, 5, 4, 4, 9, 13]
print('Median: ', compute_median(X))

X = [171, 176, 155, 167, 169, 182]
print(compute_std(X))

X = np.asarray([-2, -5, -11, 6, 4, 15, 9])
Y = np.asarray([4, 25, 121, 36, 16, 225, 81])
print('Correlation: ', compute_correlation_coefficient(X, Y))


# Tabular Data Analysis
data = pd.read_csv('advertising.csv')
X = data['TV']
Y = data['Radio']
corr_xy = correlation(X, Y)
print(f'Correlation between TV and Radio: {round(corr_xy, 2)}')

features = ['TV', 'Radio', 'Newspaper']
for feature_1 in features:
    for feature_2 in features:
        corr_val = correlation(data[feature_1], data[feature_2])
        print(
            f'Correlation between {feature_1} and {feature_2}: {round(corr_val, 2)}')

X = data['Radio']
Y = data['Newspaper']
result = np.corrcoef(X, Y)
result_2 = correlation(X, Y)
print(result)
print(result_2)

print(data.corr())

plot_heatmap(data)


# Text Retrieval
file_path = 'vi_text_retrieval.csv'
context, vi_data_df = load_data(file_path)
tfidf_vectorizer, context_embedded = tfidf_vectorize(context)
print(context_embedded)
print(context_embedded.toarray()[7][0])

question = vi_data_df.iloc[0]['question']
results = tfidf_search(question, tfidf_vectorizer, context_embedded, top_d=5)
print(results[0]['cosine_score'])
for result in results:
    print('\nId: ', result['id'])
    print('Score: ', result['cosine_score'])
    print(vi_data_df.iloc[result['id'], 1])

results = corr_search(question, tfidf_vectorizer, context_embedded, top_d=5)
print(results[1]['corr_score'])
