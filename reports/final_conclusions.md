# Final conclusions about models metrics


## Binary classification with ANN Keras. Whether a new customer will make a second purchase within 90 days of their first transaction or not

### Class 0: precision 0.98, recall 0.69, f1 0.81
- The model very rarely makes mistakes when predicting class 0, but only 69% of cases of this class are correctly captured. A decent balanced metric.

### Class 1: precision 0.03, recall 0.41, f1 0.05
- Essentially all predictions of class 1 are false. Useless f1 result.

### Accuracy: 0.68 i ROC_AUC: 0.54
- Almost a random model.

### Finally, the model is useless. This results from the class distribution 9152 vs 184 samples. XGBoost would most likely perform better, but the model serves a demonstrative function.

## Applied:
- Feature engineering and preprocessing: standardization of numerical features + OneHotEncoder for categorical ones,
- Class balancing: SMOTENC applied (oversampling considering categorical variables),
- Hyperparameter tuning: KerasTuner RandomSearch used to select number of neurons, dropouts and learning rate,
- Regularization: Dropout in two layers,
- Loss function: BinaryFocalCrossentropy (gamma=2, alpha=0.75) – handles imbalanced data well,
- Optimizer: Adam with different LR options,
- AUC as validation metric in tuner (more important than accuracy for imbalanced classes).


## Daily Reveneu forecasting with Prophet

### MAE: 7612
- Average absolute error at the level of 7,600 BRL, where daily amounts reach 60k. This is a deviation level that must be taken into account.

### MSE: 81,4 million
- Grows strongly due to large deviations.

### RMSE: 9023
- Average squared error at the level of 9k.

### MAPE: 25,6%
- The average forecast deviates by one quarter from the actual values.

### Finally, the model captures the trend, but accuracy is unsatisfactory.

## Applied:
- Custom class with parameters:
    * changepoint_prior_scale=0.3 → greater trend flexibility,
    * seasonality_prior_scale=10 → stronger seasonality fitting,
    * default yearly and weekly seasonality enabled.
- Validation split,
- Training the final model on full data and saving to file,
- Generating forecasts for 30/60/90 days and plots.


## Sentiment analysis with BERTimbau

### Class 0: precision 0.85, recall 0.78, f1 0.81
- Very good results, the model detects this class well.

### Class 1: precision 0.27, recall 0.53, f1 0.36
- Initially the model had low recall and rarely correctly captured this class. The reason was the low number of occurrences of this class in the data. After introducing weighting, recall doubled, and f1-score went up from 0.28 → 0.36. The model much more often correctly captures the rare class, though at the cost of precision.

### Class 2: f1 0.91
- Very good result.

### Macro avg: f1 0.70
- acceptable balance between classes.

### Accuracy: 0.81
- After introducing weighting, accuracy dropped but this is natural, because accuracy with imbalanced data was “masking” weak results of class 1. Now the model realistically treats all classes better, at the cost of misclassifying some samples from dominant classes.

### Finally, the model is more balanced.

## Applied:
- Pretrained model: neuralmind/bert-base-portuguese-cased (BERTimbau),
- Data balanced with weights: CrossEntropyLoss with weights inversely proportional to class frequencies,
- Scheduler: get_linear_schedule_with_warmup,
- Optimizer: AdamW with LR=2e-5,
- Mixed precision training (torch.cuda.amp.autocast, GradScaler). Speed-up and lower memory consumption,
- DataLoader with pin_memory + num_workers=4 for faster batching.