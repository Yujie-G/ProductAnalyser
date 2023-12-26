# ProductAnalyser

A project about analysing user comments of products for improving the product.

## Installation

1. Install Python 3.6+ and pip.
```bash
pip install -r requirements.txt
```

2. Download the dataset.
Download datasets from [here](https://github.com/SophonPlus/ChineseNlpCorpus/raw/master/datasets/online_shopping_10_cats/online_shopping_10_cats.zip)

3. Download the base model.
Download models from [here](https://drive.google.com/open?id=1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY)

4. Set the path of the dataset and the base model.
Manual set the path of the model and dataset in each file. Set the save path of output file.



## Usage

```bash
python SentimentAnalysis/sentiment_analysis.py
```


## Function Description

| File Path | Function Description |
| --- | --- |
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\ThemeExtraction.py | A program used to extract themes from descriptive text. |
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\autotest_scripts\config_test.py | Provides configuration parameters for automated test scripts. |
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\future_work\AspectExtraction.py | A program used to extract aspect terms from text. |
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\SentimentAnalysis\config.py | Contains configuration parameters for sentiment analysis program. |
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\SentimentAnalysis\dataset.py | Loads and processes the dataset required for sentiment analysis model.|
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\SentimentAnalysis\sentiment_analysis.py | A program that performs sentiment analysis. |
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\SentimentAnalysis\train.py | A script used to train the sentiment analysis model. |
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\SentimentAnalysis\utils.py | Contains utility functions required for sentiment analysis program. |
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\visualize\classSentiment.py | A script that visualizes sentiment analysis results. |
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\visualize\themePie.py | A script that generates the distribution of product theme weights.|
| ProductAnalyser-main.zip.extract\ProductAnalyser-main\visualize\wordCloud_fig.py | A script that generates word cloud figures. |

Overall, this project is used for analyzing product themes, sentiment, and visualizing related information.
