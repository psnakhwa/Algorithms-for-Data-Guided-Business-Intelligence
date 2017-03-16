import sys
import collections
from collections import Counter
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords = list(stopwords)
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE    
    def remove_stopwords(train_data):
        new_list=[]
        for words in train_data:
            temp_list = []
            for word in words:
                if word not in temp_list:
                    temp_list.append(word)
                    new_list.append(word)
        return new_list

    pos_dict = Counter(remove_stopwords(train_pos))
    neg_dict = Counter(remove_stopwords(train_neg))
    features = []

    for key,val in pos_dict.iteritems():
        if key not in features is True and key not in stopwords is True:
            if (val >= 0.01*len(train_pos) or neg_dict.get(key) >= 0.01*len(train_neg) and val >= 2*neg_dict.get(key)):
                features.append(key)

    for key,val in neg_dict.iteritems():
        if key not in features and key not in stopwords:
            if val >= 0.01*len(train_neg) or neg_dict.get(key) >= 0.01*len(train_neg) and val >= 2*neg_dict.get(key):
                features.append(key)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = map(lambda Y: map(lambda X: 1 if X in Y else 0, features), train_pos)
    train_neg_vec = map(lambda Y: map(lambda X: 1 if X in Y else 0, features), train_neg)
    test_pos_vec  = map(lambda Y: map(lambda X: 1 if X in Y else 0, features), test_pos)
    test_neg_vec  = map(lambda Y: map(lambda X: 1 if X in Y else 0, features), test_neg) 
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    def labelReviews(reviews,label_type):
        labelled = []
        for i in range(len(reviews)):
            label = label_type + str(i)
            labelled.append(LabeledSentence(reviews[i],[label]))
        return labelled

    labeled_train_pos = labelReviews(train_pos, 'TRAIN_POS')
    labeled_train_neg = labelReviews(train_neg, 'TRAIN_NEG')
    labeled_test_pos = labelReviews(test_pos, 'TEST_POS')
    labeled_test_neg = labelReviews(test_neg, 'TEST_NEG')

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE

    def extract_feature_vector(reviews,label_type):
        extracted_feature_vecs = []
        for i in range(len(reviews)):
            label = label_type + str(i)
            extracted_feature_vecs.append(model.docvecs[label])
        return extracted_feature_vecs

    train_pos_vec = extract_feature_vector(train_pos,'TRAIN_POS')
    train_neg_vec = extract_feature_vector(train_neg,'TRAIN_NEG')
    test_pos_vec = extract_feature_vector(test_pos,'TEST_POS')
    test_neg_vec = extract_feature_vector(test_neg,'TEST_NEG')

    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X = train_pos_vec+train_neg_vec
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=None).fit(X,Y)
    lr_model = sklearn.linear_model.LogisticRegression().fit(X,Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X = train_pos_vec+train_neg_vec
    nb_model = sklearn.naive_bayes.GaussianNB().fit(X,Y)
    lr_model = sklearn.linear_model.LogisticRegression().fit(X,Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    predicted_positive = model.predict(test_pos_vec)
    predicted_negative = model.predict(test_neg_vec)

    tp = sum(predicted_positive == "pos")
    fp = sum(predicted_negative == "pos")
    tn = sum(predicted_negative == "neg")
    fn = sum(predicted_positive == "neg")

    accuracy = (tp+tn)*(1.0)/(tp+tn+fp+fn)

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
