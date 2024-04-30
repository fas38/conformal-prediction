## Conformal Prediction

| Use Case | Dataset | TS Type | Library | Model | CP Method | Efficiency (interval width) | Validity (missclassification error) | Reproducibility | Comments | Implementation | Original Source |
|---|---|---|---|---|---|---|---|---|---|---| ---|
| Forecasting Temperature | [yosemite_temps](https://github.com/ourownstory/neuralprophet-data/blob/main/datasets/yosemite_temps.csv) | Univariate | NeuralProphet | [Autoregressive](https://neuralprophet.com/tutorials/tutorial04.html) | [CQR](https://neuralprophet.com/how-to-guides/feature-guides/uncertainty_quantification.html#Option-2:-Conformalized-Quantile-Regression) | 13.32 | 0.08 | NA |  | [temp-forecasting](https://github.com/fas38/conformal-prediction/blob/main/temperature_forecasting.ipynb) | [source](https://neuralprophet.com/how-to-guides/feature-guides/uncertainty_quantification.html#) |
| Forecasting Temperature | [yosemite_temps](https://github.com/ourownstory/neuralprophet-data/blob/main/datasets/yosemite_temps.csv) | Univariate | TorchCP | MLP | CQR | 28.19 | 0.1 | NA | depending on the training epoch coverage can reach 100 percentage | [temp-forecasting](https://github.com/fas38/conformal-prediction/blob/main/torchcp.ipynb) | [source](https://github.com/ml-stat-Sustech/TorchCP/blob/master/examples/time_series.py) |
| Forecasting Electricity Load | [SF_hospital_load](https://github.com/ourownstory/neuralprophet-data/blob/main/datasets/energy/SF_hospital_load.csv) | Univariate | NeuralProphet | Autoregressive | CQR | 76.19 | 0.02 | Yes |  | [load-forecasting](https://github.com/fas38/conformal-prediction/blob/main/neural_prophet.ipynb) | [source](https://neuralprophet.com/how-to-guides/feature-guides/uncertainty_quantification.html#) |
| Forecasting Electricity Load (1 day in future) | [ER_Europe_subset_10nodes](https://github.com/ourownstory/neuralprophet-data/blob/main/datasets/multivariate/ER_Europe_subset_10nodes.csv) | Multivariate | NeuralProphet | Autoregressive | [Naive](https://neuralprophet.com/how-to-guides/feature-guides/uncertainty_quantification.html#Option-1:-Naive-Conformal-Prediction) | 65.94 | 0.08 | NA |  | [load-forecasting](https://github.com/fas38/conformal-prediction/blob/main/multi_var_energy_load_forecast_single_step.ipynb) | [source](https://neuralprophet.com/how-to-guides/application-examples/energy_tool.html) |
| Forecasting Electricity Load (7 days in future)| [ER_Europe_subset_10nodes](https://github.com/ourownstory/neuralprophet-data/blob/main/datasets/multivariate/ER_Europe_subset_10nodes.csv) | Multivariate | NeuralProphet | Autoregressive | Naive | 483.48 | 0.08 | NA |  | [load-forecasting](https://github.com/fas38/conformal-prediction/blob/main/multi_var_energy_load_forecast_multi_step.ipynb) | [source](https://neuralprophet.com/how-to-guides/application-examples/energy_tool.html) |


## EHR Dataset

mimic iii demo (1.4) - [documentation](https://mimic.mit.edu/docs/iii/demo/)

mimic iii demo (1.4) benchmark - [source code for creating benchmark dataset](https://github.com/yerevann/mimic3-benchmarks) | [paper](https://www.nature.com/articles/s41597-019-0103-9)

    - each patient info in separate directory containing: stay, diagnose, event and events as time series
    - data split into test and train 
    - time series data for following tasks: In-hospital mortality prediction, Decompensation prediction, Length of stay prediction, Phenotype classification.
    
Practice Fusion Kaggle 2012 Dataset - [source](https://github.com/yasminlucero/Kaggle/tree/master)

