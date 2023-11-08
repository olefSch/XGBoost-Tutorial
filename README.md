# XGBoost

---

University project which describes XGBoost with an example Dataset :)

We will make a MD for Philipp because of Knowledge Mangement 🥰

## Concept for the project

1. Data Analysis and Visualization (25%)
    - Philipp Meyer
    - search for state of the art structure for eda
    - includes Data Cleaning !!!
    - finding variables

> Requirements: - Comprehensive analysis of the collected data, including statistical insights or observations. - Use of appropriate visualizations to represent training progress, performance, or other relevant aspects. - Descriptive annotations accompanying charts/graphs to provide clear interpretations. - Effective communication of the obtained insights and observations from the analysis.

2. Implementation (30%)
    - Ole Schildt
    - Why has it such a high percentage?
    - Does it include the explaination etc?
    - scikit learn XGBoost classifier but mention Regressor
    - connection to experimental setup needed

> Requirements: - Correct implementation of the chosen machine learning algorithm using the selected framework. - Clear and well-organized code structure. 3. Experimental setup (20%)

    - Ole Schildt
    - Is something like MLFlow needed for tracking
    - evaluate which metrics will we need
    - some visualizations to make all happy
    - hyperparameter tuning

> Requirements: - Thorough exploration of hyperparameter settings, including appropriate ranges. - Proper tracking and logging of training progress using relevant metrics. - Adequate number of experiments conducted to support meaningful analysis.

4. Comparision (15%)
    - Janis Hahn
    - explain CatBoost and LGMB in a overview
    - short implementation
    - tuning needed?
    - same metrics for evaluation of course
    - conclusion in MD table

> Requirements: - Inclusion of a comparison between multiple algorithms. - Clear explanation of the compared algorithms and their respective performances. - Thorough analysis of the comparison results, highlighting strengths and weaknesses of each algorithm. - Insightful discussion on the implications of the comparison and its relevance to the project topic.

5. Conclusion and Future Directions (10%)
    - together
    - should it be included in PP or/and Notebook

> Requirements: - Clear and concise summary of the main findings and insights obtained from the project. - Thoughtful discussion on the limitations of the implementation and potential improvements. - Overall coherence and logical flow of the conclusion section.

6. Documentation and Presentation (10%)
    - together
    - are medium etc okay as reference?
    - PP with mathematical explaination?

> Requirements: - Well-documented Jupyter Notebook with clear explanations and comments. - Proper organization of sections, including introduction, methodology, results, and conclusion. - Clarity of language, grammar, and overall readability. - Appropriate use of references, citations, and acknowledgments.

## Data set

-   Kaggle API!!! to avoid storing data somewhere
-   [Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)

#### Context

Reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals and is a key indicator of human progress. The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births.

Parallel to the notion of child mortality is, of course, maternal mortality, which accounts for 295,000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.

In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost-accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions, and more.

#### Data

This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetricians into 3 classes:

-   **Normal**
-   **Suspect**
-   **Pathological**

## Other Points

...
