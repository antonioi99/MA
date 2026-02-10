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
1
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
without exposure to explanations. If explanations are useful, the LLM should demonstrate
improved accuracy when provided with explanations.
Different prompting strategies have been used, and different explanation NLP formats
are presented to the LLM. The question is not only whether explanations help, but also
which explanation format and prompting strategy increase accuracy the most.
The explanation methods to be evaluated are LIME (Ribeiro et al., 2016) and SHAP
(Lundberg and Lee, 2017), with the possibility of integrating additional feature-based
methods1. The LLM-as-a-judge is “Unbabel/M-Prometheus-3B” (Pombal et al., 2025)2