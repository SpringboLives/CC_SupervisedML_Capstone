### Project Description

Using a dataset of user information provided by OKCupid and Codecademy, formulate questions about the data that can be explored
using machine learning classification and regression algorithms. Generate new columns within the dataset as needed to test your questions.

### Methods and Python modules used

- Python Modules
    - Pandas - for viewing and manipulating data (creating new columns, sorting, etc.)
    - matplotlib - for plotting and visualizing results
    - sklearn - for accessing classification and regression modules
- Classification Algorithms
    - Decision Tree
    - Random Forest
    - K Nearest Neighbor
- Regression Algorithms
    - Linear Regression
    - Multi Linear Regression

### Data Exploration
- Navigate to Visualizations > explorations.png for my charts
    - Age is right skewed
    - Sex is made up of 59.77% male and 40.23% female
    - Income features an inverted bell, but 48442 (80.81%) users reported income as -1 (this might be OKCupid's way of showing a NaN)


### Questions Asked

1. Can body type be predicted with different types of classification algorithms by looking
at diet, drinking, drug use, education, and income level?

2. Is education level an accurate way of predicting income?

I arrived at these questions after exploring the different features that existed in the dataset
and asked questions to myself about what I might be able to glean from the features. 


### New Columns Created


- Education
    - Created a numeric value to associate with the different levels of education. 
    This ranks education based on the level (high school < college < masters) and 
    the level within schooling where the user is (drop out < some school < finished)
    - There is a value of "Space Camp" in the education column. I've removed this as I
    don't consider it helpful information in trying to create continuous data 
    for the education column.
    - *Note: I tried ranking the education values from low to high with no overlapping numbers (0 to 15 with 'drop out of high school'
    being the lowest, and going up by 1 with all graduate-level schools being ranked the same). This version is stored in
    feature_selection_v2.py if you'd like to see the results there.
    
```  
    work = 'working on '
    drop = 'dropped out of '
    grad = 'graduated from '
    
    education_mapping = {
    '{}high school'.format(drop): 0,
    '{}high school'.format(work): 1,
    'high school': 2,
    '{}high school'.format(grad): 3,
    '{}two-year college'.format(drop): 1,
    '{}two-year college'.format(work): 2,
    'two-year college': 3,
    '{}two-year college'.format(grad): 4,
    '{}college/university'.format(drop): 2,
    '{}college/university'.format(work): 3,
    'college/university': 4,
    '{}college/university'.format(grad): 5,
    '{}masters program'.format(drop): 3,
    '{}masters program'.format(work): 4,
    'masters program': 5,
    '{}masters program'.format(grad): 6,
    '{}med school'.format(drop): 3,
    '{}med school'.format(work): 4,
    'med school': 5,
    '{}med school'.format(grad): 6,
    '{}law school'.format(drop): 3,
    '{}law school'.format(work): 4,
    'law school': 5,
    '{}law school'.format(grad): 6,
    }
```

- Sign_refined
    - This column was meant to change all of the qualifiers following the users' signs
    so they only contained the sign itself.
    - *The original data contained phrases like "Gemini and laughing about it", "Gemini but it doesn't matter".*
    
```   
signs = ['aquarius', 'aries', 'taurus', 'gemini', 'cancer', 'leo',
         'virgo', 'libra', 'scorpio', 'sagittarius', 'capricorn', 'pisces'
         ]


df.dropna(subset=['sign'], inplace=True)
df['sign_refined'] = np.where(df['sign'].str.contains(signs[0]), signs[0],
                        np.where(df['sign'].str.contains(signs[1]), signs[1],
                        np.where(df['sign'].str.contains(signs[2]), signs[2],
                        np.where(df['sign'].str.contains(signs[3]), signs[3],
                        np.where(df['sign'].str.contains(signs[4]), signs[4],
                        np.where(df['sign'].str.contains(signs[5]), signs[5],
                        np.where(df['sign'].str.contains(signs[6]), signs[6],
                        np.where(df['sign'].str.contains(signs[7]), signs[7],
                        np.where(df['sign'].str.contains(signs[8]), signs[8],
                        np.where(df['sign'].str.contains(signs[9]), signs[9],
                        np.where(df['sign'].str.contains(signs[10]), signs[10],
                        np.where(df['sign'].str.contains(signs[11]), signs[11],
                        'No'))))))))))))
```

### Question 1: Classifier Comparison

- Predicting **Body Type** from diet, drinking, drug use, education, and income level
    - **Decision Tree**
        - Best accuracy = 0.29200652528548127  
          Best Depth = 5  
          Time to run (s) = 0.0777902603149414
    - **Random Forest**
        - Accuracy = 0.23491027732463296  
          Time to run = 0.19149017333984375
    - **K Nearest Neighbor**
        - Best Accuracy = 0.27569331158238175  
        Best Neighbor Amt. = 35  
        Time to run = 2.9052326679229736
    - **Qualitative Discussion**
        - The level of simplicity in using each of these three methods was very close,
       with Decision Tree being the easiest to implement with the given dataset. I had an
       issue getting the shape of the array to be correct in the random forest and k nearest
       neighbor algorithms. To get past this, I had to use a `.ravel()` method in the function
       definitions I'd created.
       - Decision Tree classification showed the highest level of accuracy in predicting
       body type from the selected features. The body type 'average' is the response in 24%
       of the users, so the model is performing better than expected by 4.8%
       - Random Forest seems to perform the worst in this scenario.
       - Regarding efficiency, Decision tree is the fastest algorithm in this case. I was using
       regression to find the best depth and accuracy for Decision Tree and K Nearest Neighbor.
            - Decision Tree was only iterating through `range(1, 20)`, while K Nearest Neighbor was
            iterating through `range(1,100)` when K Nearest neighbor was given a similar range to
            iterate through, it still took longer than Decision Tree (.45 seconds)

### Question 2: Regression Model Comparison

-  Predicting Income based on education level
    - **Linear Regression**
        - `model.score` = .1592
        - Time to run = 0.002
    - **Multiple Linear Regression** - using education level and essay word count
        - `model.score` = .1268
        - Time to run = 0.001
    - **Qualitative Discussion**
        - The level of simplicity (once I had created the features properly) was about equal for each
        of these regression models. The addition of graphing the regression line on the Linear Regression
        Model was a nice addition that was also very simple
        - The time it took to run each model was negligible, they both run quickly. Though this dataset is not
        full due to the amount of rows removed from `df['income_under_100k']`
        - Accuracy is not great, and it shows that there is not a very strong
        correlation between education and income, though the linear regression line does show an increase
        in income as education level increases. When adding in the essay length for the Multi Linear Regression,
        the accuracy went down, telling us that essay length doesn't have a strong correlation to income level
            - The `model.coef_` numbers for the two features were 36215.22 and 1955.43, the latter being the
            `coef_` of essay length
   
### Overall Conclusions
   
*I'd like to return to my original questions to give my conclusions:*
1. Can body type be predicted with different types of classification algorithms by looking
at diet, drinking, drug use, education, and income level?
    - The algorithm performed better than expected in this test, but the results are not significant enough
    to determine that these features are an accurate predictor of body type.
    - I'd want to tweak my numbering system for education and diet to see if that increases the accuracy
    of the model, and I'd want to have data about how much the user exercises and their hobbies.

2. Is education level an accurate way of predicting income?
    - The Linear Regression model produced a regression line that showed an increase in income
    as level of education increased, but it was not effective at actually predicting income, as the data
    has many outliers.
    - My next steps would be to further clean and inspect the income data to find a representative data set
    with less outliers. I would like to have individuals' degree focuses and their grades to determine if
    income increases with education in certain degree programs OR in students who received good grades. 
    
    
 

        

     