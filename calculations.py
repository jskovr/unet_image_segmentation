import numpy as np
from exceptions import raise_alert

# NO FILE PATHS should be used as arguments for these functions
# these functions do not return any transformed arrays, only data
# used to transform them

def calculate_f_score(gt, pred):
    """
    Calculates the f score 
    """
    tp = np.sum(np.logical_and(gt == 1.0, pred == 1.0))
    fp = np.sum(np.logical_and(gt == 0.0, pred == 1.0))
    tn = np.sum(np.logical_and(gt == 0.0, pred == 0.0))
    fn = np.sum(np.logical_and(gt == 1.0, pred == 0.0))

    if tp + fn == 0:
        raise_alert("Division by zero", "Account for this small change by making tp + fp very small.")
    precision = tp / (tp + fp)
    if tp + fn == 0:
        raise_alert("Division by zero", "Account for this small change by making tp + fn very small.")
    recall = tp / (tp + fn)

    return (2 * precision * recall) / (precision + recall)

def calculate_metrics(gt, pred, beta=1, verbose=True):
    """
    gt: npy array ground truth
    pred: npy array prediction
    beta: see f-beta score formula
    verbose: see metrics

    returns
    a dictionary with metrics
    """

    assert len(np.unique(gt)) == 2
    # assert len(np.unique(pred)) == 2
    # positive, negative, predicted positve, predicted negative                    
    p = np.count_nonzero(gt == 1.0) # POSITIVE
    n = np.count_nonzero(gt == 0.0) # NEGATIVE
    pp = np.count_nonzero(pred == 1.0) # PREDICTED POSITIVE
    pn = np.count_nonzero(pred == 0.0) # PREDICTED NEGATIVE
    tp = np.sum(np.logical_and(gt == 1.0, pred == 1.0))
    fp = np.sum(np.logical_and(gt == 0.0, pred == 1.0))
    tn = np.sum(np.logical_and(gt == 0.0, pred == 0.0))
    fn = np.sum(np.logical_and(gt == 1.0, pred == 0.0))
    assert tp + fp == pp # GOOD
    assert tn + fn == pn # GOOD
    assert tp + fn == p # GOOD
    assert tn + fp == n # GOOD
    assert tp + fp + tn + fn == int(gt.shape[0]) * int(gt.shape[1]) * int(gt.shape[2])


    """
    true positive: hit (power)
    true negative: correct rejection
    false positive: overestimation (false alarm, type I error)
    false negative: underestimation (miss, type II error)

    false positives and true negatives deal with non _____
    true positives and false negatives deal with _____

    precision: how many retrieved items are relevant?
    recall: how many relevant items are retrieved?
    """

    if verbose:  
        print("\nConfusion matrix metrics")
        print("positive:                    {}".format(p))
        print("negative:                    {}".format(n))
        print("predicted positive:          {}".format(pp))
        print("predicted negative:          {}".format(pn))
        print("true positive:               {}".format(tp))
        print("false positive:              {}".format(fp))
        print("true negative:               {}".format(tn))
        print("false negative:              {}".format(fn))




    # Metrics including error handling for 
    # precision, false discovery rate, recall, f score
    total_population =          p + n
    prevalence =                p / (p + n)
    try:
        precision =             tp / (tp + fp) # positive predicted value: how many retrieve items are relevant?
    except ZeroDivisionError:
        precision = 0
    if pp <= 0:
        precision = 0

    false_omission_rate =      fn / (tn + fn)

    try:
        false_discovery_rate =     fp / (tp + fp)
    except ZeroDivisionError:
        false_discovery_rate = 0


    negative_predicted_value = tn / (tn + fn)

    try:
        recall =               tp / (tp + fn) # (sensitivity) true positive rate: how many relevant items are selected?
    except ZeroDivisionError:
        recall = 0

    specificity =              tn / (tn + fp) # true negative rate: how many negative selected elements are truly negative?
    fall_out =                 fp / (tn + fp) # (false alarm) false positive rate: probability of false alarm
    miss_rate =                fn / (tp + fn) # false negative rate

    # True positive rate and false positive rate (recall, fall out) are important
    # if i want to grab a AUROC curve

    try:
        positive_likelihood_ratio = recall / fall_out
    except ZeroDivisionError:
        positive_likelihood_ratio = 0


    negative_likelihood_ratio = miss_rate / specificity
    diagnostic_odds_ratio = positive_likelihood_ratio / negative_likelihood_ratio
    informedness =  recall + specificity - 1
    prevalence_threshold = (np.sqrt(recall * fall_out) - fall_out) / (recall - fall_out)
    accuracy =          (tp + tn) / (tp + tn + fp + fn) # overall effectiveness
    error =             (fp + fn) / (tp + tn + fp + fn) # classification error
    jaccard_index = tp / (tp + fn + fp) # The value of warnings (IoU)
    adjusted_accuracy = 0.5 * (specificity + recall)
    # dice_similarity_coefficient = 2 * (precision * recall) / (precision + recall)

    try:
        f_score = ( (1 + beta**2) * (precision * recall) )/ ( (beta**2 *  precision) + recall)
    except ZeroDivisionError:
        f_score = 0
    if np.isnan(f_score):
        f_score = 0.0

    if pp <= 0:
        false_discovery_rate = 1.0 # check this..
        prevalence_threshold = 0.0
        diagnostic_odds_ratio = 0.0
        f_score = 0.0
    
    if verbose:
        print("total_population:            {}".format(total_population))
        print("prevalence:                  {:.3f}".format(prevalence))
        print("(TPR) recall:                {:.3f}".format(recall))
        print("(TNR) specificity:           {:.3f}".format(specificity))
        print("(FPR) fall out:              {:.3f}".format(fall_out))
        print("(FNR) miss rate:             {:.3f}".format(miss_rate))
        print("(PPV) precision:             {:.3f}".format(precision))
        print("(FOR) false omission rate:   {:.3f}".format(false_omission_rate))
        print("(FDR) false discovery rate:  {:.3f}".format(false_discovery_rate))
        print("(NPV) neg predicted value:   {:.3f}".format(negative_predicted_value))
        print("Accuracy:                    {:.3f}".format(accuracy))
        print("Adjusted accuracy:           {:.3f}".format(adjusted_accuracy))
        print("Error:                       {:.3f}".format(error))
        print("Informedness:                {:.3f}".format(informedness))
        print("Prevalence threshold:        {:.3f}".format(prevalence_threshold))
        print("Critical Success Index:      {:.3f}".format(jaccard_index))
        print("Diagnostic Odds Ratio:       {:.3f}".format(diagnostic_odds_ratio))
        print("F score:                     {:.3f}".format(f_score))
        # print("Dice similarity coeff:       {:.3f}".format(dice_similarity_coefficient))



        # print("no predicted positive values")
        # print("Error handled for precision, false discovery rate, prevalence threshold, dor, and f score")



    # return recall, specificity, fall_out, precision, f_score, accuracy
    return {"recall":recall, 
            "precision":precision, 
            "fall_out":fall_out, 
            "miss_rate":miss_rate, 
            "f_score":f_score, 
            "accuracy":accuracy}

def compute_centroid(npy, pixel_value=1.0, verbose=False):

    """
    npy: npy array
    pixel_value: pixel value to calculate the centroid from
    """

    indices = np.argwhere(npy == pixel_value)

    if indices.size == 0:
        centroid = (np.nan, np.nan, np.nan)
        raise_alert("Nan found in centroid", "Remove nan from centroid.")
    else:
        centroid = indices.mean(axis=0).astype(np.int32)  # centroid calculation
        centroid[0] = int(centroid[0])
        centroid[1] = int(centroid[1])
        centroid[2] = int(centroid[2])

    if verbose:
        print(f"Centroid is at: {centroid}")

    return centroid

def compute_experimental_uncertainty(npy):
    """
    Take a 1D np array and find the experimental uncertainty
    """
    return np.std(npy, ddof=1) / np.sqrt(len(npy))

def compute_mean_absolute_error(npy1, npy2):
    """
    npy1: npy array that is the observed value(s)
    npy2: npy array that is the predicted value(s)
    returns MAE of two numpy arrays
    """
    return np.mean(np.abs(npy1 - npy2))

def compute_mean_squared_error(npy1, npy2):
    """
    npy1: npy array that is the observed value(s)
    npy2: npy array that is the predicted value(s)
    returns MSE of two numpy arrays
    """
    return np.mean((npy1 - npy2)**2)

def compute_random_3D_vector(dw, seed=None):
    """
    compute a random 3D vector
    """
    if seed != None:
        np.random.seed(seed)

    random_vector = np.array([
        np.random.randint(-dw, dw),
        np.random.randint(-dw, dw),
        np.random.randint(-dw, dw)
    ])

    return random_vector

def compute_uniform_scale_indices(scale, shape):
    """
    Finds the uniform scaling indices of the index relative to the center of the image
    returns uniform indices for scaling from the center of an image (half indices)
    """
    shape = np.array(shape)
    scale_factor = (scale**(1/3))
    growth = shape * (scale_factor - 1)
    indices = (growth  // 2).astype(np.int32)
    # print("indices ", indices)
    return indices

def compute_random_position_in_array(npy, seed=None):
    if seed != None:
        np.random.seed(seed)

    return np.array([np.random.randint(npy.shape[0]),
                     np.random.randint(npy.shape[1]),
                     np.random.randint(npy.shape[2])])
