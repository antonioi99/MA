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

# Similarity Groups

As mentioned in the introduction, the judging LLMs will be shown 4 instances from the dev set for each instance of the test set. The grouping strategy is based on cosine similarity scores and balanced labelling (2 instances need to be positive and 2 negative). To create the similarity groups, run the following line:
```
python main_similarity_groups.py
```
This will save the groups in the `similarity_groups/similarity_groups.json` file. The following is an example of the first entry of the file:
```
{
    "0": {
        "test_instance": "this is the review with id 0",
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

# Forward Simulation

The following is an example line to run to conduct the forward simulation experiment:
```
python main_llm.py \
    --explanation_format text_scores \
    --data_size 10000 \
    --start 0 \
    --pred_order pos_neg \
    --max_new_tokens 128 \
    --prompter pairwise \
    --llm prometheus \
    --explanation shap
```
In fact, to have complete results, you need to run 9 (explanation_format) × 2 (pred_order) × 3 (llm) × 3 (explanation) = 162 different scripts.
To run them all, you can use the following line for the **prometheus** model:
```
for pred_order in "pos_neg" "neg_pos"; do for format in "baseline" "text_scores" "text_labels" "structured_text_scores" "structured_text_labels" "top_words_scores" "top_words_labels" "natural_words" "part_of_speech"; do for explanation in "shap" "lime" "attention"; do python main_llm.py --explanation_format "$format" --data_size 10000 --start 0 --pred_order "$pred_order" --max_new_tokens 128 --prompter pairwise --llm prometheus --explanation "$explanation"; done; done; done
```
and the following line for the **llama** and **qwen** models:
```
for pred_order in "pos_neg" "neg_pos"; do for format in "baseline" "text_scores" "text_labels" "structured_text_scores" "structured_text_labels" "top_words_scores" "top_words_labels" "natural_words" "part_of_speech"; do for explanation in "shap" "lime" "attention"; do for llm in "qwen" "llama"; do python main_llm.py --explanation_format "$format" --data_size 10000 --start 0 --pred_order "$pred_order" --max_new_tokens 128 --prompter single --llm "$llm" --explanation "$explanation"; done; done; done; done
```
This will save a .json file for each specific configuration with the predictions of the judging LLM at e.g. `test_results/llama/attention/single/no_chain_of_thought/neg_pos/structured_text_labels.json`.
The following is the first entry of the mentioned file:
```
[
    [
    {
        "test_id": "0",
        "prompt": "###Task Description:\n    You are given 4 examples of movie reviews with the predictions made by a sentiment classification model. The predictions are 'NEGATIVE' or 'POSITIVE'. Each example includes an explanation showing which parts of the text influenced the model's decision. \n    Your task is to analyze the model's behavior pattern and predict what the model would output for a new test review.\n\n    ###Examples from the model:\n    [DEV_EXAMPLES]\n\n    ###Test Review:\n    [TEST_INSTANCE]\n\n    ###Question:\n    Based on the model's behavior in the examples above, what would this classification model predict for the test review?\n\n    ###Answer (reply only with 'NEGATIVE' or 'POSITIVE'):\n    ",
        "full_prompt": "###Task Description:\n    You are given 4 examples of movie reviews with the predictions made by a sentiment classification model. The predictions are 'NEGATIVE' or 'POSITIVE'. Each example includes an explanation showing which parts of the text influenced the model's decision. \n    Your task is to analyze the model's behavior pattern and predict what the model would output for a new test review.\n\n    ###Examples from the model:\n    Example 1:\nReview: Excellent cast, story line, performances. Totally believable. I realize the close knit group that exemplifies the Marine Corps. But this movie brought fear to my heart. The marines let principles be damned. It seems that this film was based on real life incidents. It shows how difficult it is to go up against the establishment. Anne Heche was utterly convincing. Sam Shepard's portrayal of a gung ho Marine was sobering. And Eric Stoltz as her attorney was so deft balancing his loyalty to the Corp but also his loyalty to his client, while high above on his tightrope. He knew what his true course of action had to be. But he was pulled apart by his immersion in the Marine tradition, loyalty to the Corps above all else. I sat riveted to the TV screen. All in all I give this one a resounding 9 out of 10.\nExplanation: The model predicted POSITIVE. These words received high attention from the model when making its prediction.\n\n HIGH ATTENTION: performances. Marine this movie heart. Anne convincing. Sam else. I screen. all give this one a 9 10.\nModel's Prediction: POSITIVE\n\nExample 2:\nReview: I've watched the first 15 minutes and I can tell that there was no consultation with any military type personnel. Judith Light's charactor (an officer) has her hair down past her shoulders! One of the first officers that greets her as she walks in to the medical facility she works at is so overweight that his pant pockets gap! No - there was no military advising them on this movie. Even an ex-military enlisted could have assisted here.\nExplanation: The model predicted POSITIVE. These words received high attention from the model when making its prediction.\n\n HIGH ATTENTION: personnel. Judith Light's (an shoulders! One overweight gap! No - military this movie. Even an here.\nModel's Prediction: NEGATIVE\n\nExample 3:\nReview: Jack Webb is riveting as a Marine Corp drill instructor in the D.I.. Webb play Sgt.Jim Moore, a tough but fair Marine whose job it is to prepare young teens for possible combat. No one could have played this role any better that Jack Webb. As a former Marine,I can assure that this is the most accurate film dealing with basic training in the Corp. Extremely entertaining!\nExplanation: The model predicted POSITIVE. These words received high attention from the model when making its prediction.\n\n HIGH ATTENTION: Jack Webb in Webb combat. that Jack Webb. this accurate film training Corp. Extremely entertaining!\nModel's Prediction: POSITIVE\n\nExample 4:\nReview: Hey there Army Sgt. I'm sorry dude but being a SGT in the Army and being in the Army National Guard does not make you qualified to comment on a Marine movie. You are not a Marine and just because you wear a uniform doesn't mean you can relate to being a Marine. We simply are the best, we have the hardest training, yes we have big heads about ourselves, but hey when you are the best, you like to strut your stuff. I was in the Iraq invasion and in Fallujah. I fought next to soldiers. You are not \"qualified\" to say anything about my Marine Corps. I hate to be the one that starts the whole \"which branch is better\", but you have no right to say you are qualified to judge a Marine movie. Oh yeah......we are Drill Instructors.......not Drill SGT's. That's the biggest clue you have no idea about what you are talking about. Yeah we do not \"curse\" at recruits anymore. Tell me, how is cussing at someone going to make them a better Marine? How will me hitting someone make a Marine a better Marine? Yes it is a kinder boot camp from what I went through. But we are dealing with different times and people. We are training people who are over all smarter than our generations recruits. We want smarter recruits, not meaner. And anyone who signs up to be a Marine in the first place, has a dedication to be the best his country has to offer. We don't have to reinforce that in Bootcamp. Marines come to Bootcamp wanting to be killers. We don't need to teach them that by demoralizing them by swearing at them and beating them. At least that is how I feel.And yes, I am \"qualified\" to say that. I have been on the battlefield numerous times and I have trained Marines and Recruits who eventually ended up on the battlefield. But then again, what do I know. I was just there, done that, got the t-shirt. SGT of the Army.......get a clue!\nExplanation: The model predicted POSITIVE. These words received high attention from the model when making its prediction.\n\n HIGH ATTENTION: Army Marine movie. doesn't movie. about. don't Marines feel.And know. SGT clue!\nModel's Prediction: NEGATIVE\n\n\n    ###Test Review:\n    When the opening shot is U.S. Marines seriously disrespecting the U.S. flag, a movie has a tough road ahead, but unfortunately it was downhill from there. There is a military adviser credited, who is also apparently a retired U.S. Marine, making it even more baffling that this incredible breach of protocol, and law, went unnoticed. Even more baffling is the way they simply glossed over how a Marine is reported KIA, then buried, in very short order, without the slightest explanation of how they identified the body, or if there even was a body. The U.S. government is still finding the missing from WWII, and it takes months to identify the remains. Military shot down remain MIA for months or years and are only declared KIA when the remains have been positively identified, or after years of red tape. Here we are expected to believe that it happens within a matter of days or weeks. Maybe this happens in Denmark, but not in the U.S. Clearly none of the people involved ever had the slightest involvement with, or respect for, the U.S. military.<br /><br />Beyond that, there are a number of other utterly laughable moments when characters come up with zingers out of nowhere. There must have been some really extended meetings between auteur and actors as they struggled to find their motivation for such hogwash. Having a script that worked might have helped, but this one seems to have been made up on the spot, working from Cliffs Notes. There's no way to know if the script was this awful originally, or if it was the auteur, or the middle-management kids at the studio who bear responsibility. Either way, this is an awful movie that should have never been made.\n\n    ###Question:\n    Based on the model's behavior in the examples above, what would this classification model predict for the test review?\n\n    ###Answer (reply only with 'NEGATIVE' or 'POSITIVE'):\n    ",
        "llm_response": "NEGATIVE",
        "predicted_label_LLM": 0,
        "dev_predictions": [
            1,
            0,
            1,
            0
        ],
        "config": {
            "use_explanations": true,
            "explanation_format": "structured_text_labels"
        }
    },
    ...
]
```