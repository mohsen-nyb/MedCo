import numpy as np
from pyhealth.metrics.multilabel import multilabel_metrics_fn
from collections import Counter
from tqdm import tqdm
import numpy as np
import scipy.stats as stats


def get_co_occurrence(data):
    co_occurrence_counts = Counter()

    for patient in tqdm(data):
        patient_label = patient['label']
        for code in patient_label:
            co_occurrence_counts[code] += 1

    return co_occurrence_counts


def get_group_labels1(data):
    '''based on frequency of codes'''
    co_occurrence_counts = get_co_occurrence(data)

    # Sort labels by their frequency counts in ascending order
    sorted_labels = [label for label, _ in co_occurrence_counts.most_common()[::-1]]

    # Calculate cumulative counts
    total_counts = sum(co_occurrence_counts.values())
    cumulative_counts = np.cumsum([co_occurrence_counts[label] for label in sorted_labels])

    # Determine the count thresholds for each group
    threshold_25 = total_counts * 0.25
    threshold_50 = total_counts * 0.50
    threshold_75 = total_counts * 0.75

    # Group labels based on cumulative counts
    groups = {'0-25': [], '25-50': [], '50-75': [], '75-100': []}
    for label, cumulative_count in zip(sorted_labels, cumulative_counts):
        if cumulative_count <= threshold_25:
            groups['0-25'].append(label)
        elif cumulative_count <= threshold_50:
            groups['25-50'].append(label)
        elif cumulative_count <= threshold_75:
            groups['50-75'].append(label)
        else:
            groups['75-100'].append(label)

    return co_occurrence_counts, groups


def get_group_labels2(data):
    '''based on number of codes'''
    co_occurrence_counts = get_co_occurrence(data)

    # Sort labels by their frequency counts in ascending order
    sorted_labels = [label for label, _ in co_occurrence_counts.most_common()[::-1]]


    # Determine the count thresholds for each group
    threshold_25 = int(len(sorted_labels) * 0.25)
    threshold_50 = int(len(sorted_labels) * 0.50)
    threshold_75 = int(len(sorted_labels) * 0.75)

    # Group labels based on cumulative counts
    groups = {'0-25': sorted_labels[0:threshold_25],
              '25-50': sorted_labels[threshold_25:threshold_50],
              '50-75': sorted_labels[threshold_50:threshold_75],
              '75-100': sorted_labels[threshold_75:]}


    return co_occurrence_counts, groups



def get_hit_at_k(y_true, y_prob, k):
    # Ensure y_true and y_prob are numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Get the indices of the top k predictions

    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]

    correct = 0
    count = 0
    for i in range(y_true.shape[0]):
        # Check if any of the true labels are in the top k predictions
        if np.sum(y_true[i, top_k_preds[i]]) > 0:
            correct += 1

        yk = np.sum(y_true[i, :])
        if yk > 0:
            count += 1

    if count == 0:
        return 0  # Avoid division by zero if no samples have true labels

    # Calculate accuracy at k
    acc_at_k = correct / count

    return acc_at_k


def get_acc_at_k2(y_true, y_prob, k):
    # Ensure y_true and y_prob are numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Get the indices of the top k predictions
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]

    # Initialize the count of correct predictions
    correct_predictions = 0

    count = 0
    for i in range(y_true.shape[0]):
        # Count how many of the true labels are in the top k predictions

        # number of labels in the patient
        yk = np.sum(y_true[i, :])
        if yk > 0:
            correct_predictions += np.sum(y_true[i, top_k_preds[i]]) / min(k, yk)
            count+=1

    if count == 0:
        return 0  # Avoid division by zero if no samples have true labels
    # Calculate accuracy at k
    acc_at_k = correct_predictions / count

    return acc_at_k


def get_group_accuracy_at_k(y_true, y_prob, co_occurrence_counts, k, group_labels):
    """
    Calculate Accuracy@k for a specific group of labels.

    Parameters:
    - y_true (np.ndarray): True labels, shape (n_samples, n_labels)
    - y_prob (np.ndarray): Predicted probabilities, shape (n_samples, n_labels)
    - co_occurrence_counts (dict): Dictionary of label counts, e.g., {'label1': count1, 'label2': count2, ...}
    - k (int): The number of top predictions to consider
    - group_labels (list): List of labels that are part of the current group

    Returns:
    - Accuracy@k for the group of labels
    """
    # Convert group_labels to a set for faster membership testing
    group_labels_set = set(group_labels)

    # Get indices of labels that are part of the current group
    group_indices = [j for j, label in enumerate(co_occurrence_counts.keys()) if label in group_labels_set]

    # Extract the true labels and predicted probabilities for the current group
    y_prob_group = y_prob[:, group_indices]
    y_true_group = y_true[:, group_indices]

    # Calculate and return accuracy at k for the current group
    return get_acc_at_k2(y_true_group, y_prob_group, k), get_hit_at_k(y_true_group, y_prob_group, k)





def get_group_eval(y_true, y_prob, co_occurrence_counts, all_metrics, group_labels):
    """
    Calculate pr-acu, roc-auc for a specific group of labels.

    Parameters:
    - y_true (np.ndarray): True labels, shape (n_samples, n_labels)
    - y_prob (np.ndarray): Predicted probabilities, shape (n_samples, n_labels)
    - co_occurrence_counts (dict): Dictionary of label counts, e.g., {'label1': count1, 'label2': count2, ...}
    - k (int): The number of top predictions to consider
    - group_labels (list): List of labels that are part of the current group

    Returns:
    - pr-acu, roc-auc for the group of labels
    """
    # Convert group_labels to a set for faster membership testing
    group_labels_set = set(group_labels)

    # Get indices of labels that are part of the current group
    group_indices = [j for j, label in enumerate(co_occurrence_counts.keys()) if label in group_labels_set]

    # Extract the true labels and predicted probabilities for the current group
    y_prob_group = y_prob[:, group_indices]
    y_true_group = y_true[:, group_indices]
    # print(f'y_prob_group0= {y_prob_group.shape}')
    # print(f'y_true_group0= {y_true_group.shape}')

    labels_to_include = []
    for i in range(y_true_group.shape[0]):
        y_true_c = y_true_group[i, :]
        # Check if there are positive samples
        if np.sum(y_true_c) > 0:
            labels_to_include.append(i)

    #print(f'labels_to_include: {labels_to_include}')

    y_prob_group = y_prob_group[labels_to_include, :]
    y_true_group = y_true_group[labels_to_include, :]
    # print(f'y_prob_group1= {y_prob_group.shape}')
    # print(f'y_true_group1= {y_true_group.shape}')

    #print(labels_to_include)

    # Calculate and return metric for the current group
    return multilabel_metrics_fn(y_true_group, y_prob_group, metrics=all_metrics)


def evaluate(y_true, y_prob, co_occurrence_counts, groups1, list_top_k, all_metrics):

    results = multilabel_metrics_fn(y_true, y_prob, metrics=all_metrics)

    all_metrics2 = [

        "pr_auc_samples",
        "roc_auc_samples"
    ]

    for group, labels in groups1.items():
        print(labels)
        results_group = get_group_eval(y_true, y_prob, co_occurrence_counts, all_metrics2, labels)
        results[f'roc_auc_samples_{group}'] = results_group['roc_auc_samples']
        results[f'pr_auc_samples_{group}'] = results_group['pr_auc_samples']
        #results[f'f1_samples_{group}'] = results_group['f1_samples']


    for top_k in list_top_k:
        acc_at_k = get_acc_at_k2(y_true, y_prob, top_k)
        hit_at_k = get_hit_at_k(y_true, y_prob, top_k)
        results[f'acc_at_k={top_k}'] = acc_at_k
        results[f'hit_at_k={top_k}'] = hit_at_k
        for group, labels in groups1.items():
            results[f'Group_acc_at_k={top_k}@' + group], results[f'Group_hit_at_k={top_k}@' + group] = get_group_accuracy_at_k(y_true, y_prob, co_occurrence_counts, top_k, labels)

    return results



def calculate_confidence_interval(performance_metrics_list, confidence_level=0.95):
    """
    Calculate the confidence interval and margin of error for a list of performance metrics.

    Parameters:
    performance_metrics_list (list): A list of performance metric values.
    confidence_level (float): The confidence level for the interval (default is 0.95 for 95%).

    Returns:
    tuple: A tuple containing the mean performance, the confidence interval (lower bound, upper bound),
           and the margin of error.
    """
    # Calculate mean and standard deviation
    mean_performance = np.mean(performance_metrics_list)
    std_performance = np.std(performance_metrics_list, ddof=1)  # Use ddof=1 for sample standard deviation

    # Number of samples
    n = len(performance_metrics_list)

    # Z-score for the given confidence level
    z = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate the margin of error
    margin_of_error = z * (std_performance / np.sqrt(n))

    # Calculate the confidence interval
    #confidence_interval = (mean_performance - margin_of_error, mean_performance + margin_of_error)

    return margin_of_error