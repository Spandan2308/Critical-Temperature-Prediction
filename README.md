## Critical Temperature

It is the temperature below which the resistance of a superconductor drops to zero. The given dataset has around 80 features and the task is to predict Tc using the XGBoost algorithm.

## Data

The data contains 21,263 unique superconductors. The given Tc values are highly right-skewed. Hence, BoxCox transformation was first applied to distribute the Tc values normally.

A Mutual information-based feature selection method was used to select the top 8 contributing features. 

An XGBoost algorithm with the following set of parameters was trained on the training dataset.

```python
parameters = {
    "objective": "reg:squarederror",
    "tree_method": "exact",
    "grow_policy": "lossguide",
    "max_depth": 6,
    "reg_alpha": 4,
    "learning_rate": 0.1,   
    "eval_metric": "rmse",
    "seed": 42
}
```

## Error Scores 
The MAE score is 0.967 and the R2 score is 0.878. The skewness of Tc seems to affect the prediction efficiency even after applying the transformation. 
