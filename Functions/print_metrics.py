def print_metrics(train_preds, train_labels, test_preds, test_labels,):
    """
    Simple function for printing metrics for ML classification score.
    
    Parameters:
    train_preds: predicted score for X_train
    train_labels: labels for y_train
    test_preds: predicted score for X_test
    test_labels: labels for y_test
    
    Returns:
    Training + Testing scores for: precision, recall, accuracy, F1, average-precision.
    """
    
    print("Training Model:")
    print("--"*8)
    print(" Precision Score (Train): {:.4}%".format((precision_score(train_labels, train_preds))*100))
    print(" Recall Score (Train): {:.4}%".format((recall_score(train_labels, train_preds))*100))
    print(" Accuracy Score (Train): {:.4}%".format((accuracy_score(train_labels, train_preds))*100))
    print(" F1 Score (Train): {:.4}%".format((f1_score(train_labels, train_preds))*100))
    print(" mAP (Train) Score: {:.4}%".format((average_precision_score(train_labels, train_preds))*100))
    print("\nTest Model:")
    print("--"*8)
    print(" Precision Score (Test): {:.4}%".format((precision_score(test_labels, test_preds))*100))
    print(" Recall Score (Test): {:.4}%".format((recall_score(test_labels, test_preds))*100))
    print(" Accuracy Score (Test): {:.4}%".format((accuracy_score(test_labels, test_preds))*100))
    print(" F1 Score (Test): {:.4}%".format((f1_score(test_labels, test_preds))*100))
    print(" mAP (Test) Score: {:.4}%".format((average_precision_score(test_labels, test_preds))*100))