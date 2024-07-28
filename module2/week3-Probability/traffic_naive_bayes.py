import numpy as np

def compute_priors(data, labels):
    return {label: np.sum(data[:, -1] == label) / data.shape[0] for label in labels}

def compute_conditional_prob(data, feature_idx, feature_value, label):
    labels = data[:, -1]
    count = np.sum((data[:, feature_idx] == feature_value) & (labels == label))
    total = np.sum(labels == label)
    return count / total

def compute_likelihoods(data, event, labels):
    likelihoods = {}
    for label in labels:
        feature_likelihoods = [compute_conditional_prob(
            data, i, event[i], label) for i in range(len(event))]
        likelihoods[label] = np.prod(feature_likelihoods)
    return likelihoods


if __name__ == '__main__':
    traffic_data = np.array([
        ['Weekday', 'Spring', 'None', 'None', 'On Time'],
        ['Weekday', 'Winter', 'None', 'Slight', 'On Time'],
        ['Weekday', 'Winter', 'None', 'None', 'On Time'],
        ['Holiday', 'Winter', 'High', 'Slight', 'Late'],
        ['Saturday', 'Summer', 'Normal', 'None', 'On Time'],
        ['Weekday', 'Autumn', 'Normal', 'None', 'Very Late'],
        ['Holiday', 'Summer', 'High', 'Slight', 'On Time'],
        ['Sunday', 'Summer', 'Normal', 'None', 'On Time'],
        ['Weekday', 'Winter', 'High', 'Heavy', 'Very Late'],
        ['Weekday', 'Summer', 'None', 'Slight', 'On Time'],
        ['Saturday', 'Spring', 'High', 'Heavy', 'Cancelled'],
        ['Weekday', 'Summer', 'High', 'Slight', 'On Time'],
        ['Weekday', 'Winter', 'Normal', 'None', 'Late'],
        ['Weekday', 'Summer', 'High', 'None', 'On Time'],
        ['Weekday', 'Winter', 'Normal', 'Heavy', 'Very Late'],
        ['Saturday', 'Autumn', 'High', 'Slight', 'On Time'],
        ['Weekday', 'Autumn', 'None', 'Heavy', 'On Time'],
        ['Holiday', 'Spring', 'Normal', 'Slight', 'On Time'],
        ['Weekday', 'Spring', 'Normal', 'None', 'On Time'],
        ['Weekday', 'Spring', 'Normal', 'Heavy', 'On Time']
    ])

    test_event = ['Weekday', 'Winter', 'High', 'Heavy']
    labels = ['On Time', 'Late', 'Very Late', 'Cancelled']

    priors = compute_priors(traffic_data, labels)
    likelihoods = compute_likelihoods(traffic_data, test_event, labels)
    posterior = {label: likelihoods[label] * priors[label] for label in labels}

    for label, prob in posterior.items():
        print(f'P("Class" = "{label}" | X) = {prob:.4f}')
