# Indian Market Gap Prediction

This project aims to create a model which will be able to predict if the next day opening price of the Indian market will be higher, lower or at the same level as current day's closing price. 

This project is useful for anyone who is interested in predicting the direction of the Indian market opening price. This can be useful for investors, traders, or anyone who wants to make informed decisions about their investments in the Indian market.

This project is non-trivial because it involves dealing with complex financial data, which can be noisy and non-stationary. Additionally, the project requires the use of machine learning algorithms to make predictions, which can be challenging to implement and interpret. Furthermore, the project requires a deep understanding of the underlying market dynamics and the factors that influence the opening price of the Indian market.

# Scope

The scope of this project is to create a probabilistic model which can predict the direction of the opening price of the Indian market for the next day. This project does not aim to explain why the market is moving in a particular direction, nor does it aim to provide any insights into the underlying market dynamics. The model will only make predictions based on the data provided to it, and will not be able to provide any explanations or justifications for its predictions.


# Architecture

The major components of this project are:

* Data Ingestion: This component is responsible for collecting and processing the data from various sources. This includes downloading the historical data from the API, cleaning and preprocessing the data, and storing it in a database.

* Data Storage: This component is responsible for storing the processed data in a database. This database will be used to train and test the machine learning model.

* Machine Learning Model: This component is responsible for training and testing the machine learning model. This model will take the processed data as input and output a prediction for the direction of the opening price of the Indian market for the next day.

* API: This component is responsible for providing an interface for the user to interact with the model. This includes providing an interface for the user to input the current market data, and receiving the prediction from the model.

The data flows through the system in the following way:

* The data is collected from various sources and stored in a database.
* The data is then processed and cleaned, and stored in a database.
* The machine learning model is trained and tested using the processed data.
* The API is used to interact with the model, providing an interface for the user to input the current market data and receive the prediction from the model.
* The model makes a prediction based on the input data and returns it to the API.
* The API then returns the prediction to the user.


# Data Sources

The data sources used in this project are:

| Type of Data | Required Fields | Frequency | Risks / Limitations | Example Thinking |
| --- | --- | --- | --- | --- |
| Daily Market Price Data | Accurate open and close prices | Daily | Any delay or missing trading day impacts label accuracy. | "This system relies on daily market price data with accurate open and close prices. Any delay or missing trading day impacts label accuracy." |
| Real-time News Data | Relevant news articles from various sources | Real-time | The data is limited to news articles that are available online, and may not include any other sources of news such as television or radio. | "This system relies on real-time news data from various sources. Any delay or missing news article impacts model accuracy." |


# Prediction Definition

The prediction made by the model is based on the concept of a "gap up", "gap down", or "gap flat". A gap up occurs when the opening price of the market is higher than the previous day's closing price, a gap down occurs when it is lower, and a gap flat occurs when it is the same. The prediction is calculated by comparing the current market data to the historical data and identifying patterns and trends that are indicative of a gap up, gap down, or gap flat.

Thresholds are used to define what constitutes a gap up, gap down, or gap flat. These thresholds are used to determine the magnitude of the difference between the opening and closing prices that is required to classify the prediction as a gap up, gap down, or gap flat. For example, a threshold of 0.5% may be used, so that if the opening price is more than 0.5% higher than the previous day's closing price, it is classified as a gap up. The use of thresholds helps to reduce the noise in the data and improve the accuracy of the predictions.


# Repository Structure

The repository is structured as follows:

* `data`: This folder contains the data processing logic, including scripts to download and clean the data, and to store it in a database. This folder should not depend on the API layer, and should only contain logic related to data processing.
* `api`: This folder contains the API logic, including the interface for the user to interact with the model. This folder should not depend on the data processing layer, and should only contain logic related to the API.
* `model`: This folder contains the machine learning model, including the code to train and test the model. This folder should not depend on the API layer, and should only contain logic related to the model.
* `utils`: This folder contains utility functions, including functions to connect to the database and to process the data. This folder should not depend on the API layer, and should only contain logic related to utility functions.
* `docs`: This folder contains the documentation for the project, including the README and any other relevant documents.
* `tests`: This folder contains the tests for the project, including unit tests and integration tests. This folder should not depend on the API layer, and should only contain logic related to testing.


# How to Run

# Environment Setup

The environment setup is currently incomplete, but it is planned to be fully documented in the future. The plan is to provide a step-by-step guide on how to set up the environment, including the installation of dependencies and the configuration of the database.

For now, the following placeholders are provided as a starting point:

* Install Python 3.7 or higher
* Install the required dependencies using pip
* Configure the database using the provided scripts
* Set up the environment variables for the API and model

More information will be provided in the future as the environment setup is completed and documented.


# Limitations and Assumptions

The system can fail in the following ways:

* Market behavior: The system assumes that the Indian market opening price will follow similar patterns as it has in the past. However, unexpected changes in market behavior can cause the system to fail.
* News ambiguity: The system relies on news articles to provide context to the data. However, ambiguous or contradictory news articles can cause the system to fail.
* Data latency: The system assumes that the data will be available in a timely manner. However, delays in data availability can cause the system to fail.
* Model decay: The system assumes that the machine learning model will remain accurate over time. However, changes in market behavior or data distribution can cause the model to decay, leading to inaccurate predictions.