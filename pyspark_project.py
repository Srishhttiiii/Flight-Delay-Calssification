# convert this code into streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Flight Delay Prediction App')
st.write('This app predicts the flight delay based on the input parameters')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load data into the dataframe.
flights = pd.read_csv('flights_small.csv', header=True)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load data into the dataframe.
airports = pd.read_csv('airports.csv', header=True)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load data into the dataframe.
planes = pd.read_csv('planes.csv', header=True)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


# Create a text element and let the reader know the data is loading.

# create function to convert data into model_data

def convert_data(flights, airports, planes):
    flights = flights.withColumnRenamed('faa', 'dest')
    model_data = flights.join(airports, on='dest', how='leftouter')
    model_data = model_data.withColumnRenamed('year', 'plane_year')
    model_data = model_data.join(planes, on='tailnum', how='leftouter')
    model_data = model_data.withColumn('arr_delay', model_data.arr_delay.cast('integer'))
    model_data = model_data.withColumn('air_time', model_data.air_time.cast('integer'))
    model_data = model_data.withColumn('month', model_data.month.cast('integer'))
    model_data = model_data.withColumn('plane_year', model_data.plane_year.cast('integer'))
    model_data = model_data.withColumn('plane_age', model_data.year - model_data.plane_year)
    model_data = model_data.withColumn('is_late', model_data.arr_delay > 0)
    model_data = model_data.withColumn('label', model_data.is_late.cast('integer'))
    model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")
    return model_data

model_data = convert_data(flights, airports, planes)

# create function to do one hot encoding, string indexer and vector assembler

def one_hot_encoding(model_data):
    from pyspark.ml.feature import StringIndexer, OneHotEncoder
    from pyspark.ml.feature import VectorAssembler
    # Create a StringIndexer
    carr_indexer = StringIndexer(inputCol='carrier', outputCol='carrier_index')
    # Create a OneHotEncoder
    carr_encoder = OneHotEncoder(inputCol='carrier_index', outputCol='carr_fact')
    # encode the dest column just like you did above
    dest_indexer = StringIndexer(inputCol='dest', outputCol='dest_index')
    dest_encoder = OneHotEncoder(inputCol='dest_index', outputCol='dest_fact')
    vec_assembler = VectorAssembler(inputCols=['month', 'air_time', 'carr_fact', 'dest_fact', 'plane_age'],
                                    outputCol='features', handleInvalid="skip")
    from pyspark.ml import Pipeline
    flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])
    piped_data = flights_pipe.fit(model_data).transform(model_data)
    return piped_data

piped_data = one_hot_encoding(model_data)

# create function to choose from 5 diff classifiers

def choose_classifier(piped_data):
    from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, NaiveBayes
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    gbt = GBTClassifier()
    nb = NaiveBayes()

    # use if else statement to choose classifier
    classifier = st.selectbox('Choose a classifier', ('Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosted Trees', 'Naive Bayes'))
    if classifier == 'Logistic Regression':
        model = lr
    elif classifier == 'Decision Tree':
        model = dt
    elif classifier == 'Random Forest':
        model = rf
    elif classifier == 'Gradient Boosted Trees':
        model = gbt
    else:
        model = nb

    # Create the evaluator
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')

    # do hyperparameter tuning for all classifiers
    if classifier == 'Logistic Regression':
        grid = ParamGridBuilder().addGrid(lr.regParam, np.arange(0, .1, .01)).addGrid(lr.elasticNetParam, [0, 1]).build()
    elif classifier == 'Decision Tree':
        grid = ParamGridBuilder().addGrid(dt.maxDepth, [2, 4, 6, 8, 10]).build()
    elif classifier == 'Random Forest':
        grid = ParamGridBuilder().addGrid(rf.maxDepth, [2, 4, 6, 8, 10]).build()
    elif classifier == 'Gradient Boosted Trees':
        grid = ParamGridBuilder().addGrid(gbt.maxDepth, [2, 4, 6, 8, 10]).build()
    else:
        grid = ParamGridBuilder().addGrid(nb.smoothing, [0, 1]).build()

    # Create the CrossValidator
    cv = CrossValidator(estimator=model, estimatorParamMaps=grid, evaluator=evaluator)

    # Fit cross validation models
    models = cv.fit(piped_data)

    # Extract the best model
    best_model = models.bestModel

    # Use the model to predict the test set
    test_results = best_model.transform(piped_data)

    # Evaluate the predictions
    print(evaluator.evaluate(test_results))

    return best_model

    