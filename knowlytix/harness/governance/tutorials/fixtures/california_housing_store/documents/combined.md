# California Housing Regression Domain Knowledge

## Dataset Overview

The California Housing dataset contains 20640 samples and 8 features.
The target variable MedHouseVal represents median house value in hundreds of thousands of dollars.
The dataset was collected from the 1990 California census.

Features include MedInc (median income), HouseAge, AveRooms (average rooms per household), AveBedrms (average bedrooms per household), Population, AveOccup (average occupancy), Latitude and Longitude.

## Feature Descriptions

| Feature | Description | Range | Type |
|---|---|---|---|
| MedInc | Median income in block group | 0.5 to 15.0 | Continuous |
| HouseAge | Median house age in block group | 1 to 52 | Continuous |
| AveRooms | Average rooms per household | 0.8 to 141.9 | Continuous |
| AveBedrms | Average bedrooms per household | 0.3 to 34.1 | Continuous |
| Population | Block group population | 3 to 35682 | Continuous |
| AveOccup | Average household occupancy | 0.7 to 1243.3 | Continuous |
| Latitude | Block group latitude | 32.5 to 42.0 | Continuous |
| Longitude | Block group longitude | -124.3 to -114.3 | Continuous |

## Feature Importance Relationships

MedInc is the strongest predictor of MedHouseVal because income directly determines purchasing power.
Latitude predicts MedHouseVal because coastal areas command higher prices.
Longitude predicts MedHouseVal because proximity to the coast increases value.
AveRooms correlates positively with MedHouseVal because larger homes cost more.
AveBedrms correlates positively with MedHouseVal because bedroom count reflects home size.
Population has weak predictive power for MedHouseVal because density effects are nonlinear.
AveOccup has moderate predictive power for MedHouseVal because overcrowding signals lower-value areas.
HouseAge has moderate predictive power for MedHouseVal because older homes may be in established neighborhoods.

## Regression Models

### LinearRegression

| Data Profile | Recommended Model | Rationale |
|---|---|---|
| Linear relationships | LinearRegression | Optimal under linearity assumption |
| Need interpretability | LinearRegression | Direct coefficient interpretation |
| Baseline comparison | LinearRegression | Standard benchmark for regression tasks |

LinearRegression fits a linear relationship between features and target.
LinearRegression assumes linear relationships between predictors and response.
LinearRegression provides direct coefficient interpretation for each feature.
LinearRegression is sensitive to multicollinearity among correlated features.
LinearRegression does not regularize coefficients and may overfit with many features.
LinearRegression achieves moderate R2 on California Housing due to nonlinear patterns.

### Ridge

| Data Profile | Recommended Model | Rationale |
|---|---|---|
| Correlated features | Ridge | L2 regularization handles multicollinearity |
| Prevent overfitting | Ridge | Shrinks coefficients toward zero |

Ridge regression adds L2 regularization to LinearRegression to prevent overfitting.
Ridge prevents overfitting by shrinking coefficients toward zero without eliminating features.
Ridge is appropriate when features are correlated like MedInc and AveRooms.
Ridge retains all features but reduces their influence on the prediction.
The alpha parameter controls regularization strength in Ridge regression.
Ridge typically achieves similar R2 to LinearRegression on California Housing.

### Lasso

| Data Profile | Recommended Model | Rationale |
|---|---|---|
| Feature selection | Lasso | L1 regularization drives coefficients to zero |
| Sparse models | Lasso | Produces interpretable sparse solutions |

Lasso regression adds L1 regularization to LinearRegression for automatic feature selection.
Lasso performs automatic feature selection by driving some coefficients to exactly zero.
Lasso is appropriate for high-dimensional sparse problems requiring interpretability.
Lasso produces sparse models that identify the most important features.
The alpha parameter controls sparsity in Lasso regression.
Lasso may drop geographic features on California Housing if alpha is too large.

### RandomForestRegressor

| Data Profile | Recommended Model | Rationale |
|---|---|---|
| Nonlinear patterns | RandomForestRegressor | Captures complex feature interactions |
| Robustness needed | RandomForestRegressor | Ensemble averaging reduces variance |
| Outlier tolerance | RandomForestRegressor | Tree splits are robust to outliers |

RandomForest builds an ensemble of decision trees using bootstrap samples for robustness.
RandomForest handles nonlinear relationships without manual feature engineering.
RandomForest is robust to outliers and noisy features in the training data.
RandomForest provides feature importance via permutation or impurity decrease.
RandomForest does not extrapolate beyond the training data range which limits generalization.
RandomForest typically achieves R2 above 0.8 on California Housing.

### GradientBoostingRegressor

| Data Profile | Recommended Model | Rationale |
|---|---|---|
| Maximum accuracy | GradientBoostingRegressor | Sequential error correction maximizes fit |
| Complex interactions | GradientBoostingRegressor | Captures high-order feature interactions |

GradientBoosting builds trees sequentially where each tree corrects previous errors.
GradientBoosting captures complex nonlinear interactions between features.
GradientBoosting achieves the highest accuracy on tabular regression tasks like housing.
GradientBoosting uses shrinkage (learning rate) for regularization to prevent overfitting.
GradientBoosting is sensitive to the number of estimators and learning rate settings.
GradientBoosting typically achieves the best R2 on California Housing among standard models.

## Regression Evaluation Metrics

| Metric | Range | Good Threshold | Interpretation |
|---|---|---|---|
| R2 | -inf to 1.0 | above 0.7 | Proportion of variance explained |
| MSE | 0 to inf | dataset-dependent | Average squared error |
| MAE | 0 to inf | dataset-dependent | Average absolute error |
| RMSE | 0 to inf | dataset-dependent | Root mean squared error |

R2 measures the proportion of variance explained by the regression model.
R2 ranges from negative infinity to 1.0 where 1.0 indicates perfect prediction.
R2 above 0.7 indicates good explanatory power for California Housing data.
R2 above 0.8 indicates strong model performance on this regression task.
MSE measures average squared prediction error and penalizes large deviations.
MAE measures average absolute prediction error and is robust to outliers.
RMSE is the square root of MSE and has the same units as the target variable.

## Model Comparison Expectations

GradientBoosting typically outperforms LinearRegression on California Housing.
RandomForest typically outperforms LinearRegression on California Housing.
Ridge and Lasso perform similarly to LinearRegression on this dataset.
Tree-based models capture the nonlinear relationship between income and price.
Linear models miss the geographic clustering pattern in California Housing.
GradientBoosting and RandomForest achieve R2 above 0.8 on California Housing.
LinearRegression achieves R2 around 0.6 on California Housing.

## Robustness Considerations

Feature perturbation of MedInc has the largest effect on model predictions.
Geographic feature perturbation affects tree-based models more than linear models.
Models trained on California Housing should be tested with feature noise injection.
Robust models maintain R2 within 10 percent under 5 percent feature perturbation.
Tree-based models are generally more robust to feature noise than linear models.

## Fairness Considerations

California Housing data reflects historical housing patterns and income inequality.
Geographic features (Latitude, Longitude) may encode demographic composition.
Models using geographic features should be audited for disparate impact.
Income-based predictions may perpetuate existing wealth disparities.
Fairness audits should compare predictions across geographic regions.
