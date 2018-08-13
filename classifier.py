import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from feature_extraction import *
import time
import pickle
import os


def build_classifier(car, non_car, path=None):

    # Create an array stack of feature vectors
    X = np.vstack((car, non_car)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car)), np.zeros(len(non_car))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Scaling features
    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    # Classifier
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    data = {"model": svm, "X_scaler": X_scaler, "X_test": X_test, "y_test": y_test}
    if path and not os.path.isfile(path):
        pickle.dump(data, open(path, 'wb'))

    return svm, X_test, y_test, X_scaler


def parse_args():
    parser = argparse.ArgumentParser(description='Parser for classifier creation extraction')
    parser.add_argument('-f', '--features', dest="fpath", help='Location of features')
    parser.add_argument('-m', '--model',    dest="mpath", help='Location of model and data')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    t1 = time.time()
    if not os.path.isfile(args.fpath):
        car, non_car = extract_all_features('./vehicles', './non-vehicles')
        save_features(car, non_car)
    else:
        data = pickle.load(open(args.fpath, 'rb'))
        car = data["car"]
        non_car = data["non-car"]

    if not os.path.isfile(args.mpath):
        svm, X_test, y_test, X_scaler = build_classifier(car, non_car, 'train.p')
    else:
        train = pickle.load(open(args.mpath, 'rb'))
        svm = train["model"]
        X_test = train["X_test"]
        y_test = train["y_test"]

    # Timing
    t2 = time.time()
    print(svm.score(X_test, y_test), t2-t1)
    exit()
