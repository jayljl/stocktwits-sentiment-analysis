import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV


# GridSearchCV ML model K-fold training - Multinomial Naive Bayes
def train_model_naive_bayes(training_data):
   
    # Creating a train test set for 500k labelled comments to train the model using a Moltuinomial NB classifier
    x_train, x_test, y_train, y_test = train_test_split(training_data['Message'], training_data['Sentiment'], 
                                                        test_size=0.2, stratify=training_data['Sentiment'])

    # Create pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB())  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    # This is where we define the values for GridSearchCV to iterate over
    parameters = {<INSERT OWN PARAMS>}

    # Do 10-fold cross validation for each of the 6 possible combinations of the above params
    grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
    grid.fit(x_train, y_train)
    
    return grid, x_test, y_test


# GridSearchCV ML model K-fold training - Logistic Regression
def train_model_logistic_regression(training_data):
   
    # Creating a train test set for 500k labelled comments to train the model using a Moltuinomial NB classifier
    x_train, x_test, y_train, y_test = train_test_split(training_data['Message'], training_data['Sentiment'], 
                                                        test_size=0.2, stratify=training_data['Sentiment'])

    # create pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', LogisticRegression())  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    # this is where we define the values for GridSearchCV to iterate over
    parameters = {<INSERT OWN PARAMS>)}

    # do 10-fold cross validation for each of the 6 possible combinations of the above params
    grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
    grid.fit(X_train, y_train)
    
    return grid, x_test, y_test


# Results & Classification Report
# GridSearch Results
def display_best_result(grid):
    
    print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
    print('\n')
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))


# Classification report for test set
def display_classification_report(df, grid, y_test, x_test):
   
    print('Test Set Classification Report')
    y_preds = grid.predict(x_test)
    print('accuracy score: ', accuracy_score(y_test, y_preds))
    print('\n')
    print('confusion matrix: \n', confusion_matrix(y_test, y_preds))
    print('\n')
    print(classification_report(y_test, y_preds))

    # Classification report for remaining data
    print('Remaining Data Set Classification Report')
    y_data = df["Message"]
    y_preds = grid.predict(y_data)
    print('accuracy score: ', accuracy_score(df["Sentiment"], y_preds))
    print('\n')
    print('confusion matrix: \n', confusion_matrix(df["Sentiment"], y_preds))
    print('\n')
    print(classification_report(df["Sentiment"], y_preds))


def run_model(model):
    if model == "NB":
        grid, x_test, y_test = train_model_naive_bayes(training_data)
        display_best_result(grid)
        display_classification_report(df, grid, y_test, x_test)
        return grid
    elif model == "LR":
        grid, x_test, y_test = train_model_logistic_regression(training_data)
        display_best_result(grid)
        display_classification_report(df, grid, y_test, x_test)
        return grid
    else:
        print('Input either:\n1. "NB" - Naive Bayes\n2. "LR" - Logistic Regression')
    

# Run
if __name__ == '__main__':
    df = pd.read_pickle("AAPL_Cleaned.pkl")
    df = df[["Sentiment", "Message"]]
    df = df[df["Sentiment"].isin(["Bullish", "Bearish"])]  # Filter down into labelled comments

    # Under-sampling 30k of bullish, 30k of bearish to fix imbalance dataset
    bullish_df = df[df["Sentiment"] == "Bullish"].sample(30000)
    bearish_df = df[df["Sentiment"] == "Bearish"].sample(30000)
    training_data = pd.concat([bullish_df, bearish_df]).sample(frac=1)

    run_model("LR")
