# Introduction

The research adopts the forward simulation framework proposed by Hase and Bansal
(2020), which tests whether explanations help a judge (in this case, an LLM) better
predict model behavior.
The task used to test the LLM is a sentiment classification task with only two labels
(positive and negative) on the IMDB dataset (Maas et al., 2011). The classifier is a fine-
tuned model from the Llama family (“yash3056/Llama-3.2-1B-imdb”). For this research,
we subdivided the test set (25k instances) of our dataset into two parts, preserving the
balance between positive and negative labels: a development set (15k instances) and a
test set (10k instances).

For experimental purposes, each instance of the test set is grouped with the four most
similar instances from the development set (2 positive and 2 negative labeled) based on
text similarity.
The experimental setup consists of two phases:
1. Learning phase: For each test instance, the LLM observes the input (text), ex-
planation in NLP format, and output (predicted label of the classification model)
of the four development set instances associated with that test review.
2. Prediction phase: The LLM predicts the output for the test instance by seeing
only its text. The memory of the LLM is reset and the experiment is repeated for
all instances of the test set.
The experiment is also conducted without showing any feature-based explanation to
the LLM (only input and output for each instance) to establish a baseline that indicates
whether explanations are genuinely helpful in understanding a black-box model.
The quality of explanations is measured by comparing prediction accuracy with and
without exposure to explanations (McNemar's binomial test). 
If explanations are useful, the LLM should demonstrate
improved accuracy when provided with explanations.


# Classifier Predictions

To get classifier ("yash3056/Llama-3.2-1B-imdb") predictions, run the following lines:
```
python main_classification_model.py --split dev
python main_classification_model.py --split test
```
This will save the dev and test set predictions together with the golden labels in separate .json files in the `classification_model_prediction` folder. These predictions are necessary to calculate LLM-as-a-judge accuracy.

# Explanations (SHAP, LIME, ATTENTION)

To get explanations, run the following lines:
```
python main_explanations.py --type shap --subset_size 15000 --start 0 --set dev
python main_explanations.py --type lime --subset_size 15000 --start 0 --set dev
python main_explanations.py --type attention --subset_size 15000 --start 0 --set dev
```
This will save the different explanations for the dev set in separate .pkl files in the `explanations/pkl` folder.
To convert the explanations into NLP format, run the following lines, for example:
```
python main_explanations.py --type formatter --subset_size 7500 --start 0 --set dev
python main_explanations.py --type formatter --subset_size 7500 --start 7500 --set dev
```
This will convert the explanations into 8 different NLP formats, saving them in separate .json files in the `explanations/NLP_format` folder. To conduct the experiment later, you will need just a single file containing all converted explanations (i.e., a file that merges all the files in the `explanations/NLP_format` folder). Run the following line to produce a single .json file containing every type of explanation for each instance in the dev set:
```
python main_explanations.py --type merge
```
This will save all explanations in NLP format in the file `explanations/NLP_format/merged_data/merged_data.json`.

# Similariy Groups

As mentioned in the introduction, the judging llms will be shown 4 instances from the dev set for each instance of the test set. The grouping strategy is based on cosine similariy scores and balanced labelling (2 instances need to be positive and 2 negative). To create the similariy groups, run the following line:
```
python main_similarity_groups.py
```
This will save the groups in the `similarity_groups/similarity_groups.json` file. This is the first entry of the file e.g.:
```
{
    "0": {
        "test_instance": "this is the review number with id 0",
        "dev_group": [
            12709,
            342,
            8599,
            8613
        ],
        "dev_predictions": [
            1,
            0,
            1,
            0
        ]
    },
    ...
}
```