# Language Translation Model

This project implements a Transformer-based model for language translation between English and Chinese. It includes scripts for training, evaluating, and testing the model.
This model is based on the original Transformer architecture described in the paper 'Attention Is All You Need' by Ashish Vaswani, 
Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Łukasz Kaiser, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin.
<br>
<br>

## Files

- `Train.py`: Main script for training the model.
- `Transformer.py`: Contains all classes to create the Transformer model.
- `en_dataset.txt`: Text file containing English sentences for training.
- `zh_dataset.txt`: Corresponding Chinese sentences for training.
- `en_test.txt`: Test set of English sentences for evaluating the model using METEOR.
- `zh_test.txt`: Corresponding Chinese sentences for evaluating the model using METEOR.
- `ScoreModel.py`: Script to evaluate the trained model using METEOR score.
- `TestModel.py`: Script to interact with the trained model for translations.
- `TL_0.810-VL_0.955_state_dict.pth`: Saved state dict of trained model.
<br>
<br>

## Performance 
Using the METEOR scoring metric from nltk.translate, the provided model received a score of 0.6138. 
It has a fairly accurate understanding of words encountered during training. 
Although some translations would be off, the overall semantic is convey the majority of the time.
<br>
The model is able to translate sentences such as 
<br>
<br>
Source:        John, do you like cats or dogs?
<br>
Translation:   John,你喜欢猫还是狗吗?
<br>
<br>
Source:        Do you want to eat dinner?
<br>
Translation:   你想吃晚饭吗?
<br>
<br>
With a fair degree of accuracy. It is also able to distinguish between words of different meaning such as, 
<br>
<br>
Source:        Do you like my new watch?
<br>
Translation:   你喜欢我的新手表吗?
<br>
<br>
Source:        When do you want to watch the movie?
<br>
Translation:   你想什么时候看电影?
<br>
<br>

## Usage

To train the model, run the Train.py script. Ensure that the datasets are correctly formatted and aligned.
<br>
Each line in the tgt dataset should match the src sentence in the corresponding line. 
<br>
<br>
To evaluate the trained model using METEOR score, run the ScoreModel.py script.
<br>
Originally intended to use BLEU, however due to certain limitations METEOR is overall a better metric to evaluate this model
<br>
<br>
To test out the model, run `TestModel.py` file, where you'll be prompted to enter the src sentence for the model to translate. 
<br>
Enter `q` to exit
<br>
<br>


## About the Model
Models are named based on their training and validation losses.
For the provided model, `TL_0.810-VL_0.955.pth`, the training loss is at 0.810 and validation loss at 0.955 after
training on the dataset for approximately 5 epochs. 
<br>


## About the Dataset
The English dataset is a modified version provided by Hugging Face's `wmt16 de-en` dataset using their `datasets` library,
<br>`from datasets import load_dataset`
<br>`load_dataset("wmt16", "de-en")`

All the English sentences were gathered together into a large text file composed of approximately 4.5M training examples. 
After applying a large number of filters the English dataset now contains around 532K sample sentences and testing dataset of around 9K sample sentences.
<br>
<br>
Filters include but are not limited to:
<br>
Removing uncommon special characters such as &, ^, @, ~, etc.
<br>
Using `flesch_kincaid_grade` from `textstat` to remove complex sentences that's beyond the scope of this model
<br>
Filters based on max-length
<br>
Filters based on min-length
<br>
Since this is originally from a de-en dataset, `langdetect` was used to remove sentences containing German words
<br>
Used a frequency counter to remove sentences that contains rarely used words, especially unique names
<br>
Filtered out duplicate sentences
<br>


## Limitations 
There are some limitations that exists within this model.
<br>
Notable examples include the difficulty in translating names, especially ones that the model has not encountered before in training. 
<br>
The model's translations are of fairly acceptable accuracy but are limited by factors such as:
1. Size of dataset.
   - Although 532K is a fair amount of training data, it can only generate translations for seen sentences that appeared within the dataset. 
   - Limitations can be seen when asked to translate topics of field-specific terminology/jargon
2. Src and Tgt languages
   - Taking into account that this model translates from English to Chinese, a METEOR score of 0.6138 is deemed acceptable for this level. <br> If the target language is one like Spanish, the overall quality would likely be better, due to similarity of western language like English and Spanish, given that they share latin roots. 
3. Tokenizers
   - In this model, a basic character-level tokenizer was used to train this model. While a sub-word level tokenizer like BPE might be more effective, a character-level tokenizer was used due to the project's scope. For longer and more complex sentences, using BPE or SentencePiece would be advisable.