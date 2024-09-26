# Analysis of Urinary Tract Infection Data Set #

# required packages
require(randomForest)
require(ROCR)
require(fastAdaboost)
devtools::install_github("sachsmc/plotROC")
require(plotROC)
require(epiR)
require(pROC)
require(tidyverse)
require(caret)
require(tableone)
library(kernlab)
library(doParallel)
cl <- makePSOCKcluster(detectCores()-1) # number of cores to use
registerDoParallel(cl)

#### Import Data Set ####
# Full Data set includes train and test data sets obtained from random sampling
# of entire data set in a 80/20 split. The original data set was obtained from
# SQL queries of the EHR and preprocessed. The full data set is separated on 
# variable "split" into "training and "validation"

urine_full <- read.csv("Data/urine_full3.csv") # file is called S1_File for manuscript
urine_train <- filter(urine_full, split == "training")
urine_val <- filter(urine_full, split == "validation")

urine_train <- select(urine_train, -split)
urine_val <- select(urine_val, -split)

# Formulas for full and reduced analyses

formula_reduced <- formula_reduced <- UCX_abnormal ~ ua_bacteria*ua_epi + ua_blood + ua_nitrite*ua_leuk + ua_wbc + dysuria + age + gender + Urinary_tract_infections

formula_full <- UCX_abnormal ~ . + ua_bacteria*ua_epi + ua_blood + ua_nitrite*ua_leuk + ua_wbc - PATID - ID - abx - dispo - UTI_diag - abxUTI


#### 
cv.ctrl <- trainControl(method = "cv",
                        number = 10, 
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        allowParallel=T)

##### XGBoost #####
tuneGridXGB <- expand.grid(
  nrounds=c(1000),
  max_depth = c(1:10),  
  eta = c(0.05, 0.1, 0.2), 
  gamma = c(0.01),
  colsample_bytree = c(0.75),
  subsample = c(0.50),
  min_child_weight = c(0)
  )

set.seed(45)
xgb_tune <-train(formula_reduced,
                 data=urine_train,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=tuneGridXGB,
                 verbose=T,
                 metric="ROC",
                 nthread =3
)

tuneGridXGB_full <- expand.grid(
  nrounds=c(1000),
  max_depth = c(10:20),  
  eta = c(0.05, 0.1, 0.2), 
  gamma = c(0.01, 0.02, 0.04),
  colsample_bytree = c(0.5,0.75),
  subsample = c(0.40, 0.50, 0.60),
  min_child_weight = c(0)
)


set.seed(45)
xgb_tune_full <-train(formula_full,
                 data=urine_train,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=tuneGridXGB_full,
                 verbose=T,
                 metric="ROC",
                 nthread =3
)


#### Support Vector Machine ####

#Train and Tune the SVM
set.seed(45)
svm_tune <- train(formula_reduced,
                  data=urine_train,
                  method = "svmLinear",
                  preProc = c("center","scale"),
                  metric="ROC",
                  tuneLength = 5,  # runs each hyperparameter to 5 values
                  trControl = cv.ctrl)

set.seed(45)
svm_tune_full <- train(formula_full,
                       data=urine_train,
                       method = "svmLinear",
                       preProc = c("center","scale"),
                       metric="ROC",
                       tuneLength = 5, # runs each hyperparameter to 5 values
                       trControl = cv.ctrl)

#### Random Forest ####
tuneGridRF <- expand.grid(.mtry=c(5:10))  # change back to 5:10

set.seed(45)
rf_tune <- train(formula_reduced,
                data = urine_train,
                method = "rf",
                metric = "ROC",
                ntree = 1000,
                tuneGrid = tuneGridRF,
                trControl = cv.ctrl)


tuneGridRF_full <- expand.grid(.mtry=c(10:20)) # change back to 10:20

set.seed(45)
rf_tune_full <- train(formula_full,
                 data = urine_train,
                 method = "rf",
                 metric = "ROC",
                 ntree = 1000,
                 tuneGrid = tuneGridRF_full,
                 trControl = cv.ctrl)

#### Elastic Net ####

tuneGridEN <- expand.grid(.alpha = seq(0, 1, 0.2),
                        .lambda = seq(0,3, 0.5))

set.seed(45)
en_tune <- train(formula_reduced,
                  data = urine_train,
                  method = "glmnet",
                  metric = "ROC",
                  tuneGrid = tuneGridEN,
                  trControl = cv.ctrl)

set.seed(45)
en_tune_full <- train(formula_full,
                 data = urine_train,
                 method = "glmnet",
                 metric = "ROC",
                 tuneGrid = tuneGridEN,
                 trControl = cv.ctrl)

#### Logistic Regression ####

set.seed(45)
glm_tune <- train(formula_reduced,
                  data = urine_train,
                  method="glm",
                  family="binomial",
                  metric = "ROC",
                  trControl = cv.ctrl)

set.seed(45)
glm_tune_full <- train(formula_full,
                  data = urine_train,
                  method="glm",
                  family="binomial",
                  metric = "ROC",
                  trControl = cv.ctrl)

#### Adaboost ####

set.seed(45)
ada_tune <- train(formula_reduced,
                 data = urine_train,
                 method="adaboost")

set.seed(45)
ada_tune_full <- train(formula_full,
                       data = urine_train,
                       method="adaboost")


#### Neural Net  #####

set.seed(45)
tuneGridnnet <-  expand.grid(size = seq(20,100, 20), 
                         decay = seq(0.1, 0.4, 0.1)) 

nnet_tune <- train(formula_reduced,
                 data = urine_train,
                 method = "nnet",
                 metric = "ROC",
                 trControl = cv.ctrl,
                 tuneGrid = tuneGridnnet)


#nnet_reduced <- nnet(formula_reduced, urine_train, size = 1, decay = 0.1, MaxNWts = 50000)

set.seed(45)
nnet_tune_full <- train(formula_full,
                        data = urine_train,
                        method = "nnet",
                        metric = "ROC",
                        trControl = cv.ctrl,
                        tuneGrid = tuneGridnnet,
                        MaxNWts = 10000)




#### Save data image ####

save.image(file = "Data/tuning.Rdata")

##### Analyis of Training Models  #####

results_reduced <- resamples(list(RF=rf_tune,
                          XGB = xgb_tune,
                          ElasticNet = en_tune,
                          Adaboost = ada_tune,
                          SVM = svm_tune,
                          GLM = glm_tune,
                          NNet = nnet_tune))

summary(results_reduced)

results_full <- resamples(list(RF=rf_tune_full,
                                  XGB = xgb_tune_full,
                                  ElasticNet = en_tune_full,
                                  Adaboost = ada_tune_full,
                                  SVM = svm_tune_full,
                                  GLM = glm_tune_full,
                                  NNet = nnet_tune_full))

summary(results_reduced)

############################ Final analysis ################################

#### Assessment of table numbers ####
# Number of encounters
nrow(distinct(urine_full, ID))

# Number of patients
nrow(distinct(urine_full, PATID))

# Number of positive/negative Urine cultures in full, training
# and validation data sets
summary(as.factor(urine_full$UCX_abnormal))
summary(as.factor(urine_train$UCX_abnormal))
summary(as.factor(urine_val$UCX_abnormal))


#################### Table 1 Generation ####################################

table_df <-dplyr::select(urine_full, age:race, insurance_status, arrival, dispo,
                         abx, UCX_abnormal, UTI_diag,
                         Abdominal_hernia:Urinary_tract_infections,
                         ANTIBIOTICS, ANTINEOPLASTICS, IMMUNOSUPPRESANT)

#table_df2 <-dplyr::select(urine_full, CVA_tenderness:polyuria, UCX_abnormal)


listvars <- c(names(dplyr::select(table_df, -UCX_abnormal)))

args_nnormal <-listvars

table1<-CreateTableOne(vars = listvars, 
                       strata =  c("UCX_abnormal"),
                       data = table_df,
                       argsNonNormal = args_nnormal)
print(table1, nonnormal=args_nnormal)
rm(listvars, agrs_nnormal, table_df)


############# Generation of Predicted Probabilities for Validation ############
probs_xgb <- predict(xgb_tune,urine_val,type = "prob")[,2]
probs_xgb_full <- predict(xgb_tune_full,urine_val,type = "prob")[,2]
probs_svm <- predict(svm_tune,urine_val,type = "prob")[,2]
probs_svm_full <- predict(svm_tune_full,urine_val,type = "prob")[,2]
probs_rf <- predict(rf_tune,urine_val,type = "prob")[,2]
probs_rf_full <- predict(rf_tune_full,urine_val,type = "prob")[,2]
probs_en <- predict(en_tune,urine_val,type = "prob")[,2]
probs_en_full <- predict(en_tune_full,urine_val,type = "prob")[,2]
probs_glm <- predict(glm_tune,urine_val,type = "prob")[,2]
probs_glm_full <- predict(glm_tune_full,urine_val,type = "prob")[,2]
probs_ada <- predict(ada_tune,urine_val,type = "prob")[,2]
probs_ada_full <- predict(ada_tune_full,urine_val,type = "prob")[,2]
probs_nnet <- predict(nnet_tune,urine_val,type = "prob")[,2]
probs_nnet_full <- predict(nnet_tune_full,urine_val,type = "prob")[,2]

probs_list <- list(probs_rf_full, probs_xgb_full, probs_xgb, probs_rf,
                   probs_ada_full, probs_ada, probs_svm_full, probs_svm,
                   probs_glm_full, probs_glm, probs_en_full, probs_en,
                   probs_nnet_full, probs_nnet)

prob_names <- c("probs_rf_full", "probs_xgb_full", "probs_xgb", "probs_rf",
                "probs_ada_full", "probs_ada","probs_svm_full", "probs_svm",
                "probs_glm_full", "probs_glm", "probs_en_full", "probs_en",
                "probs_nnet_full", "probs_nnet")

#############################  ROC plotting ####################################

roc_df<-as.data.frame(cbind(urine_val$UCX_abnormal, as.data.frame(probs_list)))

longtest <- melt_roc(roc_df, "V1", prob_names)


ggplot(longtest, aes(d = D, m = M, color = name)) + 
  geom_roc(n.cuts = 0) + 
  style_roc() + 
  ggtitle("ROC Curves for UTI Prediction Models") +
  scale_color_discrete(name="Prediction Models",
                       breaks= prob_names,
                       labels=c("Random Forest", "XGBoost", "Reduced XGBoost",
                                "Reduced Random Forest", "Adaboost",
                                "Reduced Adaboost","Support Vector Machine",
                                "Reduced Support Vector Machine",
                                "Logistic Regression",
                                "Reduced Logistic Regression","Elastic Net",
                                "Reduced Elastic Net", "Neural Net",
                                "Reduced Neural Net"))

############## Table 4 (AUC, Sens, Spec, LRs, etc.) ###############
# Data manually transfered into final tables
model_list <- list(xgb_tune, xgb_tune_full, rf_tune, rf_tune_full,
                   svm_tune, svm_tune_full, glm_tune, glm_tune_full,
                   en_tune, en_tune_full, ada_tune, ada_tune_full,
                   nnet_tune, nnet_tune_full)

# AUC calcs

auc_calc <- function(probs_list, val_data){
  t <- list()
  for (prob in probs_list){
    test_roc <- roc(urine_val$UCX_abnormal, prob)
    t1 <- as.data.frame(ci.auc(test_roc)[1:3])
    t <- c(t, t1)
  }
  t <- as.data.frame(t)
  names(t) <- prob_names
  rownames(t) <- c("lower.ci", "auc", "upper.ci")
  t
}

auc_table <- auc_calc(probs_list, urine_val)

# compare other AUCs to highest AUC
roc_high <- roc(urine_val$UCX_abnormal, probs_xgb_full)
roc_compare <- function(roc_high, val_data, probs_list){
  t <- list()
  for (prob in probs_list){
    roc_test <- roc(val_data$UCX_abnormal, prob)
    roc_p <- roc.test(roc_high, roc_test)$p.value
    t <- c(t, roc_p)
  }
  t <- as.data.frame(t)
  names(t) <- prob_names
  t
}

roc_compare_table <- roc_compare(roc_high, urine_val, probs_list)

# sens,spec,etc.
test_char_table <- function(model_list, val_data){
  t <- list()
  for (model in model_list){
    t1 <- table(predict(model,val_data), val_data$UCX_abnormal)
    t1 <- t1[c(2,1),c(2,1)] # table format is different for epi.tests
    t1 <- summary(epi.tests(t1, conf.level = 0.95))
    t <- c(t, t1)
  }
  t <- as.data.frame(t)
  rownames(t) <- c("aprev", "tprev", "se", "sp", "diag_acc", "diag_or",
                     "nnd", "youden", "ppv", "npv", "plr", "nlr")
  t
}

t4 <- test_char_table(model_list, urine_val)

###################### Table 5a and 5b Analyses ################################

# Create Admit and Discharge Specific Data Sets from Validation Data Set
utrain_disch_uti <- dplyr::filter(urine_val, dispo == "Discharge")
utrain_admit_uti <- dplyr::filter(urine_val, dispo == "Admit")
utrain_admit_uti$UCX_abnormal <- as.factor()


# Further Filter to create data sets where antibiotics given
u_admit_diag <- filter(utrain_admit_uti, abx == "Yes")
u_disch_diag <- filter(utrain_disch_uti, abx == "Yes")

epi_test_f <- function(condition, test){
  t1 <- table(test, condition)
  t1 <- t1[c(2,1),c(2,1)] # table format is different for epi.tests
  t1 <- epi.tests(t1, conf.level = 0.95)
  t1
}

# Table Data for 5a
epi_test_f(utrain_disch_uti$UCX_abnormal, utrain_disch_uti$abxUTI)
epi_test_f(utrain_admit_uti$UCX_abnormal, utrain_admit_uti$abxUTI)

# Table Data for 5b
epi_test_f(utrain_disch_uti$UCX_abnormal, utrain_disch_uti$UTI_diag)
epi_test_f(utrain_admit_uti$UCX_abnormal, utrain_admit_uti$UTI_diag)


############ Additional Data for results in manuscript #########################
# Table data of UTI diagnosis and whether antiboitics given for admitted and
# discharged patients in validation data set

table(utrain_disch_uti$UTI_diag, utrain_disch_uti$abx)
table(utrain_admit_uti$UTI_diag, utrain_admit_uti$abx)


# Table data for (+/-) urine culture and whether there was an alternative
# diagnosis given in the clinical impression
table(u_admit_diag$UCX_abnormal, u_admit_diag$alt_diag)
table(u_disch_diag$UCX_abnormal, u_disch_diag$alt_diag)

# Table data for (+/-) UTI diagnosis and whether there was an alternative
# diagnosis given in the clinical impression
table(u_admit_diag$UTI_diag, u_admit_diag$alt_diag)
table(u_disch_diag$UTI_diag, u_disch_diag$alt_diag)

rm(utrain_admit_uti, utrain_disch_uti, u_admit_diag, u_disch_diag)




