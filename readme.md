## Conformal Prediction

| Task Type | Dataset Type | Features | Library | CP Method | Coverage | Width of Interval/Predicted Set Size | Reproducibility | Comments | Implementation | Original Source |
|-----------|---------------|----------|---------|-----------|----------|--------------------------------------|------------------|----------|----------------|-----------------|
|    Binary Classification       |        Tabular [German Credit Dataset](https://www.openml.org/search?type=data&status=active&sort=runs&id=31)       |       20   |  [Nonconformist](https://github.com/donlnz/nonconformist)    |    Transductive      |    0.89     |   1.23                                   |  Yes                |    NA      |        [notebook](./tabular_classification_binary.ipynb)        |      [source](https://github.com/PacktPublishing/Practical-Guide-to-Applied-Conformal-Prediction/blob/main/Chapter_05_TCP.ipynb)           |
|    Binary Classification       |        Tabular [German Credit Dataset](https://www.openml.org/search?type=data&status=active&sort=runs&id=31)       |       20   |  [Crepes](https://crepes.readthedocs.io/en/latest/crepes_nb_wrap.html)    |    Inductive      |    0.93     |   1.40                                   |  NA                |    NA      |        [notebook](./tabular_classification_binary.ipynb)        |      [source](https://crepes.readthedocs.io/en/latest/crepes_nb_wrap.html)           |
|    Binary Classification       |        Tabular [German Credit Dataset](https://www.openml.org/search?type=data&status=active&sort=runs&id=31)       |       20   |  [Crepes](https://crepes.readthedocs.io/en/latest/crepes_nb_wrap.html)    |    Inductive (Mondrian Class-conditional)     |    0.93     |   1.40                                   |  NA                |    NA      |        [notebook](./tabular_classification_binary.ipynb)        |      [source](https://crepes.readthedocs.io/en/latest/crepes_nb_wrap.html)           |
|    Binary Classification       |        Tabular [German Credit Dataset](https://www.openml.org/search?type=data&status=active&sort=runs&id=31)       |       20   |  [Venn Abers](https://github.com/ip200/venn-abers/tree/main)    |    Venn Abers Calibration     |    NA     |   NA                                   |  NA                |    NA      |        [notebook](./tabular_classification_binary.ipynb)        |     [source-1](https://github.com/ip200/venn-abers/blob/main/src/venn_abers.py), [souce-2](https://www.kaggle.com/code/carlmcbrideellis/classifier-calibration-using-venn-abers)           |
|    Multi Class Classification       |        Tabular [Human Activity Recognition](https://www.openml.org/search?type=data&status=active&id=1478)       |       561   |  [MAPIE](https://mapie.readthedocs.io/en/stable/examples_classification/4-tutorials/plot_main-tutorial-classification.html#sphx-glr-examples-classification-4-tutorials-plot-main-tutorial-classification-py)    |    Inductive      |    1     |   1.26                                   |  NA                |    NA      |        [notebook](./tabular_classification_multi.ipynb)        |  
|    Regression       |       Univariate Timeseries [Medium Website Visit Dataset](https://raw.githubusercontent.com/marcopeix/time-series-analysis/master/data/medium_views_published_holidays.csv)       |       NA   |  [MAPIE](https://mapie.readthedocs.io/en/stable/examples_regression/4-tutorials/plot_ts-tutorial.html)    |    [EnbPI](https://proceedings.mlr.press/v139/xu21h/xu21h.pdf) and ACI (adaptive interval)     |    0.95, 0.94     |   652, 704                                   |  NA                |    NA      |        [notebook](./timeseries_univariate.ipynb)        |     NA         |               |
|    Regression       |       Univariate Timeseries [Medium Website Visit Dataset](https://raw.githubusercontent.com/marcopeix/time-series-analysis/master/data/medium_views_published_holidays.csv)       |       NA   |  [MAPIE](https://mapie.readthedocs.io/en/stable/examples_regression/4-tutorials/plot_ts-tutorial.html)    |    EnbPI and ACI (adaptive interval) with partial fitting    |    0.93, 0.93     |   583, 633                                   |  NA                |    NA      |        [notebook](./timeseries_univariate.ipynb)        |     NA         |               |
|    Regression       |       Univariate Timeseries [Medium Website Visit Dataset](https://raw.githubusercontent.com/marcopeix/time-series-analysis/master/data/medium_views_published_holidays.csv)       |       NA   |  [Neural Prophet](https://neuralprophet.com/how-to-guides/application-examples/energy_hospital_load.html)    |    [naive](https://neuralprophet.com/how-to-guides/feature-guides/uncertainty_quantification.html#Option-1:-Naive-Conformal-Prediction) and [CQR](https://neuralprophet.com/how-to-guides/feature-guides/uncertainty_quantification.html#Option-2:-Conformalized-Quantile-Regression)     |    0.88, 0.9     |   40.63, 118.70                                   |  NA                |    NA      |        [notebook](./timeseries_univariate.ipynb)        |     NA         |               |

## Time Series Forecasting

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


Sample notebook on creating resource allocation focused timeseries dataset from MIMIC III - [source](./test.ipynb)


## Time Series Forecasting
multivariate - [source](https://github.com/fas38/conformal-prediction/blob/main/ts_multi_var.ipynb)

## Binary Classification
binary classification - [source](https://github.com/fas38/conformal-prediction/blob/main/binary_clf.ipynb)

[slide](./documents/Binary%20Classification%20for%20Tabular%20Data.pptx)