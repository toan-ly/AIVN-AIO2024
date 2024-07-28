import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self.prior = {}

    def create_train_dataset(self, filepath):
        data = pd.read_csv(filepath)
        return data.to_numpy()

    def compute_prior(self, train_data):
        y_unique = ['Yes', 'No']
        prior_probability = np.zeros(len(y_unique))

        total_samples = train_data.shape[0]
        for i, label in enumerate(y_unique):
            count = np.sum(train_data[:, -1] == label)
            prior_probability[i] = count / total_samples
        self.prior = dict(zip(y_unique, prior_probability))

        return self.prior

    def compute_conditional_prob(self, train_data):
        y_unique = ['Yes', 'No']
        list_x_name = []
        conditional_prob = []

        for i in range(train_data.shape[1] - 1):
            x_unique = np.unique(train_data[:, i])
            list_x_name.append(x_unique)

            x_conditional_prob = np.zeros((len(x_unique), len(y_unique)))
    
            for j, x in enumerate(x_unique):
                for k, y in enumerate(y_unique):
                    count = np.sum((train_data[:, i] == x) & (train_data[:, -1] == y))
                    x_conditional_prob[j, k] = count / np.sum(train_data[:, -1] == y)
            conditional_prob.append(x_conditional_prob)
            
        return conditional_prob, list_x_name
    
    def train_naive_bayes(self, train_data):
        self.prior = self.compute_prior(train_data)
        conditional_prob, list_x_name = self.compute_conditional_prob(train_data)
        return self.prior, conditional_prob, list_x_name

    def predict(self, X, list_x_name, prior, conditional_prob):
        x_indices = [get_index_from_value(X[i], list_x_name[i]) for i in range(len(X))]
        p_yes = prior['Yes']
        p_no = prior['No']

        for i, x_idx in enumerate(x_indices):
            p_yes *= conditional_prob[i][x_idx, 0]
            p_no *= conditional_prob[i][x_idx, 1]

        y_pred = 1 if p_yes > p_no else 0
        return y_pred
        

def get_index_from_value(feature_name, list_features):
    return np.where(list_features == feature_name)[0][0]

def main():
    naive_bayes_cls = NaiveBayesClassifier()
    train_data = naive_bayes_cls.create_train_dataset('tennis_data.csv')
    print(train_data)

    prior = naive_bayes_cls.compute_prior(train_data)
    print(prior)
    for label, prob in prior.items():
        print(f'P(play tennis = {label}) = {prob}')

    conditional_prob, list_x_name = naive_bayes_cls.compute_conditional_prob(train_data)
    print(conditional_prob)

    print('\nx1 = ', list_x_name[0])
    print('x2 = ', list_x_name[1])
    print('x3 = ', list_x_name[2])
    print('x4 = ', list_x_name[3])
    
    outlook = list_x_name[0]
    i1 = get_index_from_value('Overcast', outlook)
    i2 = get_index_from_value('Rain', outlook)
    i3 = get_index_from_value('Sunny', outlook)
    print(i1, i2, i3)

    # Compute P("Outlook"="Sunny"|"Play Tennis"="Yes")
    x1 = get_index_from_value('Sunny', list_x_name[0])
    print('P("Outlook"="Sunny"|"Play Tennis"="Yes") = ', np.round(conditional_prob[0][x1, 0], 2))   
    print('P("Outlook"="Sunny"|"Play Tennis"="No") = ', np.round(conditional_prob[0][x1, 1], 2))   

    X = ['Sunny', 'Cool', 'High', 'Strong']
    pred = naive_bayes_cls.predict(X, list_x_name, prior, conditional_prob)
    
    if pred:
        print('Ad should go!')
    else:
        print('Ad should not go!')

if __name__ == '__main__':
    main()
