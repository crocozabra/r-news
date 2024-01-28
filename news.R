library(tidyverse)
library(tidytext)
library(syuzhet)
library(tibble)
library(plyr)
library(tm) #for cleaning
library(caret)
library(reshape2)
library(e1071) #for NB
library(randomForest) #for RF



data = read.csv("fake_or_real_news.csv")

#DATA ANALYSIS 

summary(data)
summary(data$label) 
" We have 3164  of fake news and 3171 of real news. 
This provides a balanced distribution between 
the two classes, which is good for training classification models."

#title length comparison
t_test_result = t.test(nchar(data$title[data$label == "FAKE"]),
                        nchar(data$title[data$label == "REAL"]))
"The extremely small p-value (< 0.05) suggests that there is 
a significant difference in the mean title length between fake and real news.
The positive t-value (13.249) and the 95% confidence interval that 
does not include 0 (6.643859 to 8.951389) indicate that the mean title length for fake news 
is significantly larger than the mean title length for real news."

print(t_test_result)
ggplot(data, aes(x = label, y = nchar(title), fill = label)) +
  geom_boxplot() +
  labs(title = "Title Length Comparison",
       x = "Label",
       y = "Title Length")

#sentiment analysis on the first 400 real and fake texts and pie chart visualization
real_texts = data$text[data$label == "REAL"][1:400]
fake_texts = data$text[data$label == "FAKE"][1:400]


sentiments_real_texts = get_nrc_sentiment(real_texts)
emotions_real_texts = colSums(sentiments_real_texts)
emotions_real_texts_df = data.frame(emotion = names(emotions_real_texts), count = emotions_real_texts)
head(sentiments_real_texts)


ggplot(emotions_real_texts_df, aes(x = "", y = count, fill = emotion)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +
  labs(title = "Emotions in Real News Texts (First 400)",
       fill = "Emotion")


sentiments_fake_texts = get_nrc_sentiment(fake_texts)
emotions_fake_texts = colSums(sentiments_fake_texts)
emotions_fake_texts_df = data.frame(emotion = names(emotions_fake_texts), count = emotions_fake_texts)
head(sentiments_fake_texts)


ggplot(emotions_fake_texts_df, aes(x = "", y = count, fill = emotion)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +
  labs(title = "Emotions in Fake News Texts (First 400)",
       fill = "Emotion")

"Fake news texts tend to have higher counts of anger, disgust.
Real news texts exhibit higher counts of trust and positive sentiments.
The presence of surprise in both fake and real news texts suggests 
that both types designed to capture the reader's attention. 
Fake news texts seem to evoke a wider range of emotions, 
including both positive and negative, which might align with the goal
of capturing attention or generating sensationalism."


#NEWS TYPE PREDICTION

#cleaning data 
preprocess_corpus = function(text) {

  text = tolower(text) 
  text = removePunctuation(text) 
  text = removeWords(text, stopwords("en")) 
  text = stemDocument(text) 
  text = stripWhitespace(text) 
  
  return(text)
}

data$title = sapply(data$title, preprocess_corpus)
data$text = sapply(data$text, preprocess_corpus)

data$label = as.factor(data$label)

#train and test data split (80% - training, 20 - testing )
set.seed(123)
splitIndex = createDataPartition(data$label, p = 0.8, list = FALSE)
train_data = data[splitIndex, ]
test_data = data[-splitIndex, ]

#retraining with different features

#length
train_data$title_length = nchar(train_data$title)
test_data$title_length = nchar(test_data$title)

train_data$text_length = nchar(train_data$text)
test_data$text_length = nchar(test_data$text)

#word count
train_data$title_word_count = sapply(strsplit(train_data$title, " "), length)
test_data$title_word_count = sapply(strsplit(test_data$title, " "), length)

train_data$text_word_count = sapply(strsplit(train_data$text, " "), length)
test_data$text_word_count = sapply(strsplit(test_data$text, " "), length)

#presence of numbers
train_data$text_num = grepl("\\d", train_data$text)
test_data$text_num = grepl("\\d", test_data$text)

train_data$title_num = grepl("\\d", train_data$title)
test_data$title_num = grepl("\\d", test_data$title)

#punctuation count
train_data$text_punct_count =sapply(strsplit(train_data$text, "[[:punct:]]"), length) - 1
test_data$text_punct_count = sapply(strsplit(test_data$text, "[[:punct:]]"), length) - 1

train_data$title_punct_count = sapply(strsplit(train_data$title, "[[:punct:]]"), length) - 1
test_data$title_punct_count = sapply(strsplit(test_data$title, "[[:punct:]]"), length) - 1



#detection from title using Naive Bayes Classifier
title_nb_model = naiveBayes(label ~ title + title_length + title_word_count + title_num + title_punct_count, data = train_data)
title_nb_pred = predict(title_nb_model, newdata = test_data)
title_nb_accuracy = confusionMatrix(title_nb_pred, test_data$label)$overall["Accuracy"]


print(paste("Accuracy of Naive Bayes Classifier - Title :", title_nb_accuracy))

#detection from title using Random Forest Classifier
title_rf_model = randomForest(label ~ title +  title_length + title_word_count + title_num + title_punct_count, data = train_data)
title_rf_pred = predict(title_rf_model, newdata = test_data)
title_rf_accuracy = confusionMatrix(title_rf_pred, test_data$label)$overall["Accuracy"]

print(paste("Accuracy of Random Forest Classifier - Title:", title_rf_accuracy))

#detection from text using Naive Bayes Classifier
text_nb_model = naiveBayes(label ~ text + text_length + text_word_count + text_num + text_punct_count, data = train_data)
text_nb_pred = predict(text_nb_model, newdata = test_data)
text_nb_accuracy = confusionMatrix(text_nb_pred, test_data$label)$overall["Accuracy"]


print(paste("Accuracy of Naive Bayes Classifier - Text :", text_nb_accuracy))

#detection from text using Random Forest Classifier
text_rf_model = randomForest(label ~ text + text_length + text_word_count + text_num + text_punct_count, data = train_data)
text_rf_pred = predict(text_rf_model, newdata = test_data)
text_rf_accuracy = confusionMatrix(text_rf_pred, test_data$label)$overall["Accuracy"]


print(paste("Accuracy of Random Forest Classifier - Tex :", text_rf_accuracy))

#detection using terms appearing in title or text using Naive Bayes Classifier
combined_nb_model = naiveBayes(label ~ title + text + title_length + title_word_count + title_num + title_punct_count +
                                  text_length + text_word_count + text_num + text_punct_count, data = train_data)
combined_nb_pred = predict(combined_nb_model, newdata = test_data)
combined_nb_accuracy = confusionMatrix(combined_nb_pred, test_data$label)$overall["Accuracy"]


print(paste("Accuracy of Naive Bayes Classifier - Combined (Title + Text):", combined_nb_accuracy))

#detection using terms appearing in title or text using Random Forest Classifier
combined_rf_model = randomForest(label ~ title + text + title_length + title_word_count + title_num + title_punct_count +
                                    text_length + text_word_count + text_num + text_punct_count, data = train_data)
combined_rf_pred = predict(combined_rf_model, newdata = test_data)
combined_rf_accuracy = confusionMatrix(combined_rf_pred, test_data$label)$overall["Accuracy"]


print(paste("Accuracy of Random Forest Classifier - Combined (Title + Text):", combined_rf_accuracy))



#detection from text using SVM Classifier
title_svm_model = svm(label ~ title_length + title_word_count + title_num + title_punct_count, data = train_data)
title_svm_pred = predict(title_svm_model, newdata = test_data)
title_svm_accuracy = confusionMatrix(title_svm_pred, test_data$label)$overall["Accuracy"]

print(paste("Accuracy of SVM Classifier - Title:", title_svm_accuracy))

#detection from text using SVM Classifier
text_svm_model = svm(label ~ text_length + text_word_count + text_num + text_punct_count, data = train_data)
text_svm_pred = predict(text_svm_model, newdata = test_data)
text_svm_accuracy = confusionMatrix(text_svm_pred, test_data$label)$overall["Accuracy"]

print(paste("Accuracy of SVM Classifier - Text:", text_svm_accuracy))

#detection using terms appearing in title or text using SVM Classifier
combined_svm_model = svm(label ~ title_length + title_word_count + title_num + title_punct_count +
                            text_length + text_word_count + text_num + text_punct_count, data = train_data)
combined_svm_pred = predict(combined_svm_model, newdata = test_data)
combined_svm_accuracy = confusionMatrix(combined_svm_pred, test_data$label)$overall["Accuracy"]

print(paste("Accuracy of SVM Classifier - Combined (Title + Text):", combined_svm_accuracy))


#WRITE THE TITLE HERE 
new_title = "Trump takes on Cruz, but lightly
"

#WRITE THE TEXT HERE 
new_text = "Killing Obama administration rules, dismantling Obamacare and pushing through tax reform are on the early to-do list.
"

# to predict from a new title
predict_from_title = function(title, model, features) {

  title = preprocess_corpus(title)
  

  new_data =data.frame(
    title = title,
    title_length = nchar(title),
    title_word_count = length(unlist(strsplit(title, " "))),
    title_num = grepl("\\d", title),
    title_punct_count = sum(str_count(title, "[[:punct:]]"))
  )
  

  pred = predict(model, newdata = new_data)
  
  return(pred)
}



new_title_prediction_nb = predict_from_title(new_title, title_nb_model)
new_title_prediction_rf = predict_from_title(new_title, title_rf_model)
new_title_prediction_svm = predict_from_title(new_title, title_svm_model)


print(paste("Naive Bayes Prediction for the new title:", new_title_prediction_nb, "Accuracy:", title_nb_accuracy*100, "%"))
print(paste("Random Forest Prediction for the new title:", new_title_prediction_rf, "Accuracy:", title_rf_accuracy*100, "%"))
print(paste("SVM Prediction for the new title:", new_title_prediction_svm, "Accuracy:", title_svm_accuracy*100, "%") )


# to predict from a new text
predict_from_text = function(text, model, features) {
  
  text = preprocess_corpus(text)
  
  
  new_data = data.frame(
    text = text,
    text_length = nchar(text),
    text_word_count = length(unlist(strsplit(text, " "))),
    text_num = grepl("\\d", text),
    text_punct_count = sum(str_count(text, "[[:punct:]]"))
  )
  
  
  pred = predict(model, newdata = new_data)
  
  return(pred)
}


new_text_prediction_nb = predict_from_text(new_text, text_nb_model, 
                                            c('text_length', 'text_word_count', 'text_num', 
                                              'text_punct_count'))


new_text_prediction_rf = predict_from_text(new_text, text_rf_model, 
                                            c('text_length', 'text_word_count', 'text_num', 
                                              'text_punct_count'))

new_text_prediction_svm = predict_from_text(new_text, text_svm_model, 
                                             c('text_length', 'text_word_count', 'text_num', 
                                               'text_punct_count'))


print(paste("Naive Bayes Prediction for the new text:", new_text_prediction_nb, "Accuracy:", text_nb_accuracy*100, "%"))
print(paste("Random Forest Prediction for the new text:", new_text_prediction_rf, "Accuracy:", text_rf_accuracy*100, "%"))
print(paste("SVM Prediction for the new text:", new_text_prediction_svm, "Accuracy:", text_svm_accuracy*100, "%"))





# to predict from a new title and text
predict_from_title_and_text = function(title, text, model, features) {

  title = preprocess_corpus(title)
  text = preprocess_corpus(text)
  
 
  new_data = data.frame(
    title = title,
    title_length = nchar(title),
    title_word_count = length(unlist(strsplit(title, " "))),
    title_num = grepl("\\d", title),
    title_punct_count = sum(str_count(title, "[[:punct:]]")),
    
    text = text,
    text_length = nchar(text),
    text_word_count = length(unlist(strsplit(text, " "))),
    text_num = grepl("\\d", text),
    text_punct_count = sum(str_count(text, "[[:punct:]]"))
  )
  

  pred = predict(model, newdata = new_data)
  
  return(pred)
}


combined_nb_prediction = predict_from_title_and_text(new_title, new_text, combined_nb_model, 
                                                      c('title_length', 'title_word_count', 'title_num', 'title_punct_count', 
                                                        'text_length', 'text_word_count', 'text_num', 'text_punct_count'))

combined_rf_prediction = predict_from_title_and_text(new_title, new_text, combined_rf_model, 
                                                      c('title_length', 'title_word_count', 'title_num', 'title_punct_count', 
                                                        'text_length', 'text_word_count', 'text_num', 'text_punct_count'))

combined_svm_prediction = predict_from_title_and_text(new_title, new_text, combined_svm_model, 
                                                       c('title_length', 'title_word_count', 'title_num', 'title_punct_count', 
                                                         'text_length', 'text_word_count', 'text_num', 'text_punct_count'))



print(paste("Naive Bayes Prediction for the new title and text:", combined_nb_prediction, 
            "Accuracy:", combined_nb_accuracy*100, "%"))
print(paste("Random Forest Prediction for the new title and text:", combined_rf_prediction,
            "Accuracy:", combined_rf_accuracy*100, "%"))
print(paste("SVM Prediction for the new title and text:", combined_svm_prediction, 
            "Accuracy:", combined_svm_accuracy*100, "%"))


accuracy_data = data.frame(
  Method = c("Naive Bayes - Title", "Random Forest - Title", "Naive Bayes - Text", "Random Forest - Text", "Naive Bayes - Combined", "Random Forest - Combined", "SVM - Title", "SVM - Text", "SVM - Combined"),
  Accuracy = c(title_nb_accuracy*100, title_rf_accuracy*100, text_nb_accuracy*100, text_rf_accuracy*100, combined_nb_accuracy*100, combined_rf_accuracy*100, title_svm_accuracy*100, text_svm_accuracy*100, combined_svm_accuracy*100)
)
accuracy_data

ggplot(accuracy_data, aes(x = Method, y = Accuracy, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Accuracy Rates of Different Methods", x = "Method", y = "Accuracy") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

"
