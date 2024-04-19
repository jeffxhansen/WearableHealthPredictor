# NYC_Taxi_Trip_Duration Predictor

# Dataset link

[NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview)

[Weather](https://www.kaggle.com/datasets/aadimator/nyc-weather-2016-to-2022)

[Google Distance](https://www.kaggle.com/datasets/debanjanpaul/new-york-city-taxi-trip-distance-matrix)

## Research Question (Markdown)
1. Why your question is interesting, well thought out, precisely formulated, and answerable with the data and machine learning
2. Briefly review what is already known about your research questions and what techniques others have used to study these questions. (references to prior work)
3. Explain the data set, before analysis.
4. Form a thoughtful hypothesis or hypotheses about the data. Answer the following questions and any others that may be relevant to your question and your data set:
Where did the data come from?
Why is the data set legitimate? What weaknesses or problems does the data set have?
Why is this a good choice of data set to answer your research questions?
What do you expect your analysis to reveal?
What other interesting questions will analyzing this data answer?

(1) The question we are considering for our project is: given basic information about your current situation, can you predict how long your taxi ride will be while in New York. The question of predicting taxi ride times in New York City is inherently intriguing and holds significant relevance for urban commuters and transportation systems. The research question is carefully crafted to explore the various factors that affect taxi ride times in cities, such as traffic patterns, congestion, time of day, and external elements like events or weather conditions. The careful crafting of the research question ensures a straightforward and measurable goal: predicting the anticipated duration of taxi rides. Having a clear and precise research question is essential for building models that can analyze extensive datasets containing historical taxi ride information and relevant factors. The availability of diverse data further strengthens the feasibility of answering the question, providing a solid foundation for training models to recognize patterns and make accurate predictions. The decision to use machine learning methods is fitting for addressing the intricacies of the problem, utilizing these techniques to capture detailed relationships within the data. Beyond individual convenience, predicting taxi ride times has practical implications for optimizing taxi fleet management, allocating resources efficiently, and improving overall traffic flow in the city. In summary, the research question is well-crafted and promising, offering valuable insights to enhance urban transportation efficiency and user experience through the use of data and machine learning.

(2) Since our data comes from a Kaggle competition, over one thousand other groups have investigated this research. Successful teams performed feature engineering to create fields such as month, day, hour, day of the week, haversine/manhattan distances. The used techniques including Random Forest Regression, Extra Trees Regression, PCA, XGBoost, linear regression, and Light GBM. 

(3, 4) The initial dataset comes from a Kaggle competition and a separate Kaggle dataset. The competition is to predict New York City taxi cab travel times from 2016 based primarily off of pick up and drop off locations and times. However, the dataset also includes fields for the tax driver, the number of passengers, and weather the trip time was recording in real time. The second dataset contains weather information for the same time period and is sourced directly from the Open Meteo website. The available fields include timestamps, temperature, precipitation, cloud cover, and wind information. The NYC taxi cab dataset is well documented and densely populated with over one million data points. It was published by NYC Taxi and Limousine Commission (TLC) in Big Query on Google Cloud Platform. The weather dataset is much more sparse, specially in the wind and cloud fields, and has a much larger time range than needed to match the NYC Taxi data. These datasets provide a good source for addressing our research questions because they extensively cover NYC taxi travel for a significant time period. By combining this taxi data with meteorological information, we can delve into various factors influencing trip duration in an urban landscape and employ machine learning techniques such as XGBoost. In short, the data allows us to effectively identify significant features impacting taxi trip duration and develop a robust predictive model for accurate estimations.

(4) While our primary investigation is focused on determining taxi cab trip duration, we are also interested in several other questions. What dates, days of the week, and times of day are most busy? Where are the most popular destinations? What factors influence taxi trip time the most? (1) The fusion of NYC taxi data and meteorological information to understand factors influencing trip duration in a bustling metropolis offers a compelling inquiry into urban transportation dynamics. Furthermore, the multifaceted exploration of temporal, spatial, and environmental influences on travel durations in NYC, complemented by the weather dataset, presents a well-rounded analysis to reveal comprehensive insights into travel behaviors. Our data is well suited for our primary, precise research in exploring and predicting taxi trip duration, and has a single, clear answer. The dataset allows us to employ machine learning techniques like XGBoost to effectively identify and classify the significant features that impact taxi trip duration. This process helps us go one step further and develop a robust predictive model to estimate and understand trip durations accurately.



## Data Cleaning / Feature Engineering (Code)
1. Organize and format your data.
2. Detect and correct (or remove) problems like missing, badly formatted, or incorrect data.
3. (Markdown) Justify your choices of what you removed, edited, reformatted, or left unchanged.
4. Apply various data handling techniques to engineer additional features that may be useful for answering your research questions.

## Robustness (Code)
1. Data cleaning code should be easy to modify for similar data sets, or handle more data points.

## Data Visualization and Basic Analysis (Code): 
1. Analyze the data, draw conclusions, and effectively communicate your main observations and results.
2. Calculate appropriate summary statistics.
3. Use appropriate isualizations and other tools to thoughtfully identify and evaluate what the data is telling you andhow well suited the data are to answering your problem
4. Make it pretty! (See below.)

## Learning Algorithms and In-depth Analysis (Code): 
1. Analyze the data using the techniques discussed in class.
2. (Markdown) Explain what research questions you can answer using the techniques presented this semester.
3. (Markdown) Explain the results of your analysis, whether the results are meaningful, and why you chose the tools that you used.
4. Don't falsify results

## Ethical Implications (Markdown): 
1. Analyze the ethical implications of your research questions, the data you gathered, and the analysis that was performed.
2. Are there privacy or other implications from the collection or use of the data?
3. Could your results and methods be misused or misunderstood?
4. What can and should be done to prevent misuse and misunderstanding?
5. Could your algorithms and methods result in a destructive self-fulfilling feedback loop?
6. How could that be prevented or controlled?
7. What other ethical implications does your work have?


# project outline

* py_files/data_manager.py
    * this will be the py file with functions like:
        * get_data() - that pulls in the merged csvs
        * get_cleaned_data() - handles any data cleaning
* py_files/features.py
    * this will handle all of the features logic
        * generate_features() - takes in a cleaned dataframe and adds all of the feature columns
* py_files/ml.py
    * this will handle all of the machine learning logic
        * train()
        * test()
        * create_model()
* py_files/helper_funcs.py
    * this will contain miscelaneous function that are used for many of the py files. They generally handle general-logic things
* config.py
    * all hyperparameters and constants for all of our code
        * features_toggle - a dictionary with True/False toggles that specify which features we will use
        * TRAIN_TEST_PERCENTAGE - the size of the train test split (probably going to be 0.80)
* analysis.ipynb
    * this will be where we have all of our graphs and charts that output performance for the ml model
    * this can also have some preliminary data analytics
* wh_predictor.ipynb
    * the final ipynb that we will submit as our writeup
* jeff.ipynb
    * jeff's personal notebook where they do all of their development and analysis before adding to the py files
* dylan.ipynb
    * dylan's personal notebook where they do all of their development and analysis before adding to the py files
* jason.ipynb
    * jason's personal notebook where they do all of their development and analysis before adding to the py files
* dallin.ipynb
    * dallin's personal notebook where they do all of their development and analysis before adding to the py files
