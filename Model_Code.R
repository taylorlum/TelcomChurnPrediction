library(dplyr)
library(readr)
library(caret)
library(rpart)
library(rpart.plot)

# read in the data, specify appropriate factor levels
telco <- read_csv("telco.csv", col_types = cols(
  gender = col_factor(levels = c("Male", "Female")), 
  SeniorCitizen = col_factor(levels = c("0", "1")), 
  Partner = col_factor(levels = c("No", "Yes")), 
  Dependents = col_factor(levels = c("No","Yes")), 
  PhoneService = col_factor(levels = c("No", "Yes")), 
  MultipleLines = col_factor(levels = c("No phone service", "No", "Yes")), 
  InternetService = col_factor(levels = c("No", "DSL", "Fiber optic")), 
  OnlineSecurity = col_factor(levels = c("No internet service", "No", "Yes")), 
  OnlineBackup = col_factor(levels = c("No internet service", "No", "Yes")), 
  DeviceProtection = col_factor(levels = c("No internet service", "No", "Yes")), 
  TechSupport = col_factor(levels = c("No internet service", "No", "Yes")), 
  StreamingTV = col_factor(levels = c("No internet service", "No", "Yes")), 
  StreamingMovies = col_factor(levels = c("No internet service", "No", "Yes")), 
  Contract = col_factor(levels = c("Month-to-month", "One year", "Two year")), 
  PaperlessBilling = col_factor(levels = c("No", "Yes")), 
  PaymentMethod = col_factor(levels = c("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)")), 
  Churn = col_factor(levels = c("No", "Yes"))))

# remove customer ID column and make capitalization consistent
telco <- telco %>% select(-1) %>%
  rename(Gender = gender,
         Tenure = tenure)

# replace missing total charges with 0s (have not been billed yet)
telco$TotalCharges[is.na(telco$TotalCharges)] = 0

# descriptive model: logistic regression
full_model = glm(Churn~., data = telco, family = "binomial")
summary(full_model)

mod_backward = step(full_model, direction = "backward", k = 2, trace = 0)
summary(mod_backward)

# Tenure: higher = decreases
# Internet service: having = increases
# One and two year contracts = decreases (vs month to month)
# Paperless billing: having = increases
# Payment method other than electronic check = decreases
# Total charges: higher = increases

# descriptive model: decision tree
set.seed(19)
fulltree = rpart(Churn~., data = telco, method = "class", cp = 0, minsplit = 2, minbucket = 1, maxsurrogate = 0)
prp(fulltree, type = 1, extra = 1)
printcp(fulltree)

# find the best pruned tree:
# min error tree = 0.77421 with xstd = 0.018142
# min xerror + xstd = 0.792352
bestpruned = prune(fulltree, cp = 0.007)
prp(bestpruned, type = 1, extra = 1)
printcp(bestpruned)
# contract: one year or two year = less churn
# internet service: none or DSL = less churn (vs fiber optic)
# tenure: longer = less churn
# tech support: using tech support = less churn (encourage people to reach out if they have problems)
# payment method: not a crucial factor but electronic check tends to lead to higher churn path
# multiple lines: more of an end factor too but multiple lines = churn

# split training and testing data
set.seed(19)
train_index = createDataPartition(telco$Churn, p = 0.8, list = FALSE)

train = telco[train_index,]
test = telco[-train_index,]

# predictive model: logistic regression
fullmodel = glm(Churn~., data = train, family = "binomial")
summary(fullmodel)

stepmodel = step(fullmodel, direction = "backward", k = 2, trace = 0)
summary(stepmodel)

# tune threshold then test
logmodel_prob = predict(stepmodel, type = "response")
logmodel_class = ifelse(logmodel_prob >= 0.4, 1, 2)
logmodel_class = factor(logmodel_class, labels = c("No", "Yes"))
logmodel_matrix = confusionMatrix(logmodel_class, train$Churn, positive = "Yes")
logmodel_matrix

# test
logmodel_prob = predict(stepmodel, type = "response", newdata = test)
logmodel_class = ifelse(logmodel_prob >= 0.4, 1, 2)
logmodel_class = factor(logmodel_class, labels = c("No", "Yes"))
logmodel_matrix = confusionMatrix(logmodel_class, test$Churn, positive = "Yes")
logmodel_matrix


# predictive model: decision tree
set.seed(19)
fulltree = rpart(Churn~., data = train, method = "class", cp = 0, minsplit = 2, minbucket = 1, maxsurrogate = 0)
prp(fulltree, type = 1, extra = 1)
printcp(fulltree)

# find the best pruned tree:
# min error tree = 0.78008 with xstd = 0.020334
# min xerror + xstd = 0.800414

bestpruned = prune(fulltree, cp = 0.00502)
prp(bestpruned, type = 1, extra = 1)
printcp(bestpruned)

# tune threshold then test
tree_prob = predict(bestpruned, type = "prob")[,2]
tree_class = ifelse(tree_prob >= 0.25, 2, 1)
tree_class = factor(tree_class, labels = c("No", "Yes"))
tree_matrix = confusionMatrix(tree_class, train$Churn, positive = "Yes")
tree_matrix

# test
tree_prob = predict(bestpruned, newdata = test, type = "prob")[,2]
tree_class = ifelse(tree_prob >= 0.25, 2, 1)
tree_class = factor(tree_class, labels = c("No", "Yes"))
tree_matrix = confusionMatrix(tree_class, test$Churn, positive = "Yes")
tree_matrix

# predictive model: random forest
library(randomForest)
set.seed(19)

# trained with different node sizes
rf_model <- randomForest(Churn ~ ., data = train, importance = TRUE, nodesize = 10)

# predict probabilities 
rf_train_prob = predict(rf_model, train, type = "prob")[, "Yes"]

# convert raw probabilities to a 0 or 1 class using threshold (test)
rf_train_class = ifelse(rf_train_prob >= 0.25, "Yes", "No")
rf_train_class = factor(rf_train_class, levels = c("No", "Yes"))

# confusion matrix for training data
confusionMatrix(rf_train_class, train$Churn, positive = "Yes")

# predict probabilities on test data
rf_test_prob = predict(rf_model, test, type = "prob")[, "Yes"]

# apply threshold
rf_test_class = ifelse(rf_test_prob >= 0.25, "Yes", "No")
rf_test_class = factor(rf_test_class, levels = c("No", "Yes"))

# generate confusion matrix for testing data
confusionMatrix(rf_test_class, test$Churn, positive = "Yes")

# predictive model: neural network
library(neuralnet)
library(fastDummies)

# transform the data
# convert characters/factors to dummy columns automatically
telco_nn <- dummy_cols(train, remove_selected_columns = TRUE, remove_first_dummy = TRUE)
test_nn <- dummy_cols(test, remove_selected_columns = TRUE, remove_first_dummy = TRUE)

# rename columns to ensure there are no spaces or special characters for formula compatibility
names(telco_nn) <- make.names(names(telco_nn))
names(test_nn) <- make.names(names(test_nn))

# scale numeric columns
num_cols <- c("Tenure", "MonthlyCharges", "TotalCharges")
telco_nn[num_cols] <- scale(telco_nn[num_cols])
test_nn[num_cols] <- scale(test_nn[num_cols])

# train with different hidden layers/nodes
set.seed(19)
nn_model <- neuralnet(
  Churn_Yes ~ ., 
  data = telco_nn, 
  act.fct = "logistic",
  hidden = 6, 
  linear.output = FALSE,
  stepmax = 1e6
)

# plot of network
library(NeuralNetTools)
dev.new(width = 12, height = 8)
plotnet(
  nn_model,
  circle_cex = 2.5,   
  cex_val = 0.55,          
  pad_x = 0.6,             
  pos_col = "black",       
  neg_col = "grey70",
  alpha_val = 0.4
)

# predict probabilities on validation data
valid.pred = predict(nn_model, select(test_nn, -"Churn_Yes"))

# convert raw probabilities to a 0 or 1 class using threshold 
valid.class = ifelse(valid.pred > 0.25, 1, 0)

# generate the confusion matrix using actual validation labels
table(valid.class, test_nn$Churn_Yes)

# calculate overall accuracy
mean(valid.class == test_nn$Churn_Yes)


# predictive model: SVM
library(e1071)
set.seed(19)

# find best parameter values for cost and gamma (using transformed data)
tune.out=tune(svm,as.factor(Churn_Yes)~.,data = telco_nn,kernel="radial",
              ranges=list(cost=c(0.001,0.01,0.1,1,10,100),
                          gamma = c(0.001, 0.01, 0.1, 1)))
summary(tune.out)

bestmod = tune.out$best.model
summary(bestmod) # gamma = 0.01, cost = 1

# further tune around those values
set.seed(19)
tune.out_tuned=tune(svm,as.factor(Churn_Yes)~.,data = telco_nn,kernel="radial",
              ranges=list(cost=c(0.25, 0.5, 0.75, 1, 1.25, 1.5, 2),
                          gamma = c(0.05, 0.08, 0.1, 0.2, 0.3)))

bestmod = tune.out_tuned$best.model
summary(bestmod)
# gamma = 0.05, cost = 0.5

set.seed(19)
svmfit.opt=svm(as.factor(Churn_Yes)~.,data=telco_nn, kernel="radial",
               gamma=bestmod$gamma, cost=bestmod$cost, probability=TRUE)

# predict probabilities
ypred=predict(svmfit.opt,newdata=select(test_nn, -"Churn_Yes"), probability = TRUE) 
yprob = attributes(ypred)$probabilities
churn_probabilities = yprob[, "1"] 

# use threshold to get class 
ypred_high_sensitivity = ifelse(churn_probabilities > 0.25, 1, 0)

# generate confusion matrix
table(pred = ypred_high_sensitivity, true = test_nn$Churn_Yes)
mean(ypred_high_sensitivity == test_nn$Churn_Yes)

TP <- sum(ypred_high_sensitivity == 1 & test_nn$Churn_Yes == 1)
FN <- sum(ypred_high_sensitivity == 0 & test_nn$Churn_Yes == 1)
TP / (TP + FN)

