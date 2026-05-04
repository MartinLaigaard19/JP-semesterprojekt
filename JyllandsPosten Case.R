# JP case
pacman::p_load(
  tidyverse, tidymodels, themis, table1, ggpubr, broom, ggfortify,
  GGally, PerformanceAnalytics, car, skimr, discrim, glmnet,
  kknn, naivebayes, kernlab, xgboost, gridExtra, rpart, future,
  ranger, rmarkdown, rvest, httr, jsonlite, rlist, rjson,
  Rcrawler, hrbrthemes, knitr, hms, leaps, readxl, gbm,
  randomForest, stringr, rsample, haven, lubridate, vip, dplyr, 
  janitor, yardstick)

# *******************************************************************************************
#                                        Dataforberedelse                                ----
# *******************************************************************************************
##                                      Indlæsning af data                               ----
# *******************************************************************************************
# Indlæser de 3 datasæt
behavior <- read.csv("data/behavior.csv")

cancellation <- read.csv("data/cancellation.csv")

subscription <- read.csv2("data/subscription_v2.csv")

# *******************************************************************************************
##                                    Oprydning & many-to-many                           ----
# *******************************************************************************************
# Oprydning af subscription for at undgå dobbelt  pseudo_id
# Undgår many-to-many problemet
subscription <- subscription |>  
  group_by(pseudo_id) |> 
  arrange(desc(order_date)) |> 
  slice(1) |> 
  ungroup()

# Oprydning af cancellation for at undgå dobbelt reason
# Undgår many-to-many problemet
cancellation <- cancellation |> 
  group_by(pseudo_id) |> 
  arrange(desc(expiration_date)) |> 
  slice(1) |> 
  ungroup()

master_data <- subscription |> 
  left_join(cancellation, by = c("pseudo_id"))

# *******************************************************************************************
##                                 Behavior features                                     ----
# *******************************************************************************************
# Joiner behavior data og behandler variabler
behavior_features <- behavior |>
  mutate(
    kategori = str_extract(page_url_clean, "(?<=jyllands-posten\\.dk/)([^/]+)")
  ) |>
  group_by(pseudo_id) |> 
  summarise(
    avg_scroll         = mean(scroll_depth),
    page_views         = n(),
    days_with_activity = n_distinct(dt),
    restricted_read    = sum(page_restricted == "yes", na.rm = TRUE),
    dominant_category  = as.factor(
      if (all(is.na(kategori))) NA_character_ 
      else names(which.max(table(na.omit(kategori))))
    ),
    .groups = "drop"
  )

master_data <- master_data |> 
  left_join(behavior_features, by = "pseudo_id")

# Tjekker om duplikater. 
master_data |> 
  group_by(pseudo_id) |> 
  filter(n() > 1)

# *******************************************************************************************
#                                        Nye variabler                                   ----
# *******************************************************************************************
# Konverterer datoer og laver nye variable
master_data <- master_data |> 
  mutate(
    birthdate                = as.Date(birthdate, format = "%d-%m-%Y"),
    subscription_cancel_date = as.Date(subscription_cancel_date, format = "%d-%m-%Y"),
    usr_created              = as.Date(usr_created, format = "%d-%m-%Y"),
    first_campaign_day       = as.Date(first_campaign_day, format = "%d-%m-%Y"),
    last_campaign_day        = as.Date(last_campaign_day, format = "%d-%m-%Y"),
    order_date               = as.Date(order_date),
    expiration_date          = as.Date(expiration_date),
    # Demografi
    age_at_order             = as.numeric((order_date - birthdate) / 365),
    days_since_registration  = as.numeric(order_date - usr_created),
    gender                   = as.factor(koen),
    # Har kunden tidligere deltaget i kampagner
    is_returning = as.factor(if_else(previous_subscriptions >= 1, 1, 0)),
    # Læst indhold
    pages_on_restricted      = restricted_read / page_views,
    # Vækst i nyhedsbreve
    newsletter_growth        = newsletters_after_order - newsletters_before_order,
    # Udvikling af nyhedsbreve
    permission_upgraded      = if_else(permission_given_order == "false" & permission_given_today == "true", 1, 0),
    permission_withdrawn     = if_else(permission_given_order == "true" & permission_given_today == "false", 1, 0) 
  )

# *******************************************************************************************
#                                        Target Variablet                                ----
# *******************************************************************************************
# Beregner churn baseret på expiration_date
master_data <- master_data |> 
  mutate(
    # Churner brugeren efter kampagnen
    churned = ifelse(is.na(expiration_date), 0, 1),
    # Churner bruger kort tid efter kampagne udløb
    churn_flag = case_when(
      subscription_cancel_date - last_campaign_day <= 30 ~ "early_churn",
      TRUE ~ "no_early_churn"
    )
  )

table(master_data$churned)

# Skaber overblik og årsagerne til at de churner
årsag_overblik <- master_data |> 
  filter(churned == "1") |> 
  count(reason, sort = TRUE)

print(årsag_overblik)

# *******************************************************************************************
#                                        Feature selection                               ----
# *******************************************************************************************
# Vi vælger variabler til ML modellering
model_data <- master_data |> 
  select(
    churned, age_at_order, gender, is_returning, account_active_days, permission_given_order,
    permission_given_today, previous_subscriptions, previous_campaigns, avg_scroll,
    page_views, days_with_activity, newsletter_growth, dominant_category, permission_upgraded,
    permission_withdrawn, pages_on_restricted
  )

gender_probs <- prop.table(table(model_data$gender[model_data$gender != ""]))

n_unknown <- sum(model_data$gender == "")

model_data <- model_data |> 
  mutate(
    gender = case_when(
    gender == "" ~ sample(names(gender_probs), size = n(), replace = TRUE, prob = gender_probs),
    TRUE ~ gender),
    gender = case_when(
      gender == "Mand" ~ 0, # Mand bliver til 0
      gender == "Kvinde" ~ 1 # Kvinde bliver til 1
    ),
    age_at_order = if_else(is.na(age_at_order), median(age_at_order, na.rm = TRUE), age_at_order),
    avg_scroll = if_else(is.na(avg_scroll), median(avg_scroll, na.rm = TRUE), avg_scroll),
    page_views = if_else(is.na(page_views), median(page_views, na.rm = TRUE), page_views),
    days_with_activity = if_else(is.na(days_with_activity), median(days_with_activity, na.rm = TRUE), days_with_activity),
    dominant_category = if_else(is.na(dominant_category), "other", dominant_category),
    pages_on_restricted = if_else(is.na(pages_on_restricted), median(pages_on_restricted, na.rm = TRUE), pages_on_restricted),
    dominant_category = fct_lump_prop(dominant_category, prop = 0.05, other_level = "other")
  )

# Konvetere characther til factor
model_data <- model_data |> 
  mutate(across(where(is.character), as.factor)) |> 
  mutate(
    churned = factor(churned),
    gender = factor(gender),
    permission_upgraded = factor(permission_upgraded),
    permission_withdrawn = factor(permission_withdrawn)
  )

# Reorder variablerne sådan at arbejdet i python bliver lidt nemmere
  model_data <- model_data |> 
  select(
    # Target
    churned,
    # Faktorer med 2 levels
    gender, is_returning, -permission_given_order, -permission_given_today, 
    permission_upgraded, permission_withdrawn,
    # Faktorer med mere end 2 levels
    dominant_category,
    # Numeriske
    age_at_order, account_active_days, previous_subscriptions, previous_campaigns,
    avg_scroll, page_views, days_with_activity, newsletter_growth, pages_on_restricted
  )

write_csv(model_data, "model_data.csv")

# *******************************************************************************************
#                                        Modellering                                     ----
# *******************************************************************************************
# Bruger tidymodels til at lave modellerne
# Bruger set.seed(42)
set.seed(42)
jp_split <- initial_split(model_data, prop = 0.70, strata = churned)

jp_train <- training(jp_split)
jp_test  <- testing(jp_split)

# Vi bruger bootstrapping da vi også kører downsampling i vores recipe. Lille datasæt, mere præcise resultater
jp_boots <- bootstraps(jp_train, times = 25, strata = churned)

# Vi definierer hvad der skal ske med data FØR modellen kører
jp_recipe <- recipe(churned ~ ., data = jp_train) |> 
  step_novel(all_nominal_predictors()) |>                       # håndterer nye factor-niveauer i test
  step_dummy(all_nominal_predictors()) |>                       # laver dummy-variable af faktorer (fx gender)
  step_zv(all_predictors()) |>                                  # fjerner kolonner med nul-varians
  step_normalize(all_numeric_predictors()) |>                   # skalerer numeriske variable (vigtigt for Ridge/Lasso)
  step_corr(all_numeric_predictors(), threshold = 0.75) |> 
  step_downsample(churned)                                      # Gør 1/0 er lige store (159 styk), kun på træningsdata

# Logistisk regression
glm_model_jp <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification")

# Ridge 
ridge_model_jp <- logistic_reg(penalty = tune(), mixture = 0) |> 
  set_engine("glmnet") |> 
  set_mode("classification")

# Lasso 
lasso_model_jp <- logistic_reg(penalty = tune(), mixture = 1) |> 
  set_engine("glmnet") |> 
  set_mode("classification")

# Elastic Net 
enet_model_jp <- logistic_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet") |> 
  set_mode("classification")

# LDA 
lda_model_jp <- discrim_linear() |> 
  set_engine("MASS") |> 
  set_mode("classification")

# QDA
qda_model_jp <- discrim_quad() |> 
  set_engine("MASS") |> 
  set_mode("classification")

# Naïve Bayes
nb_model_jp <- naive_Bayes() |> 
  set_engine("naivebayes") |> 
  set_mode("classification")

# Decision Tree
tree_model_jp <- decision_tree(cost_complexity = tune()) |> 
  set_engine("rpart") |> 
  set_mode("classification")

# Random Forest
rf_model_jp <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) |> 
  set_engine("ranger", importance = "impurity") |> 
  set_mode("classification")

# XGBoost
xgb_model_jp <- boost_tree(mtry = tune(), trees = 500, tree_depth = tune(), learn_rate = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

# Vi kombinerer recipe + alle modeller i ét samlet workflow
jp_wf_set <- workflow_set(
  preproc = list(jp = jp_recipe),
  models = list(
    logistisk  = glm_model_jp,
    ridge      = ridge_model_jp,
    lasso      = lasso_model_jp,
    elasticnet = enet_model_jp,
    lda        = lda_model_jp,
    naivebayes = nb_model_jp,
    tree       = tree_model_jp,
    rf         = rf_model_jp,
    xgboost    = xgb_model_jp
  )
)

# Metrics vi vil evaluere på
jp_metrics <- metric_set(roc_auc, accuracy, sensitivity, specificity, f_meas)

# Plan(multisession) bruger flere CPU-kerner så det går hurtigere, vi kører det parallelt
plan(multisession, workers = parallel::detectCores() - 1)

start.time <- Sys.time()

jp_fit <- jp_wf_set |> 
  workflow_map(
    seed       = 42,
    grid       = 10,          # Antal kombinationer der testes ved tuning
    resamples  = jp_boots,    
    metrics    = jp_metrics,
    verbose    = TRUE         # Printer fremgang undervejs
  )

print(Sys.time() - start.time)

plan(sequential)  # sluk parallelkørsel igen
show_notes(.Last.tune.result)

# Rankér modellerne efter ROC AUC
jp_fit |> 
  rank_results(rank_metric = "roc_auc", select_best = TRUE) |> 
  dplyr::select(wflow_id, .metric, mean, rank) |>
  filter(.metric %in% c("roc_auc", "accuracy", "sensitivity", "specificity", "f_meas")) |> 
  pivot_wider(names_from = .metric, values_from = mean) |> 
  arrange(rank)

# Tjekker for multikollinearitet
model_data |> 
  select(where(is.numeric)) |> 
  cor(use = "complete.obs") |> 
  round(2)

# *******************************************************************************************
#                                  Bedste model (Random Forest)                          ----
# *******************************************************************************************
# Finder ID på bedste model
bedste_jp_id <- jp_fit |> 
  rank_results(rank_metric = "roc_auc", select_best = TRUE) |> 
  dplyr::slice(1) |> 
  pull(wflow_id)

bedste_jp_id 

# Udtræk resultater og workflow for den bedste model
bedste_resultat_jp <- jp_fit |> 
  extract_workflow_set_result(bedste_jp_id)

bedste_wf_jp <- jp_fit |> 
  extract_workflow(id = bedste_jp_id)

# Vælg de bedste hyperparametre
bedste_params_jp <- select_best(bedste_resultat_jp, metric = "roc_auc")

# Finaliser workflow med de bedste parametre
final_wf_jp <- finalize_workflow(bedste_wf_jp, bedste_params_jp)

# Fit på hele træningsdata og evaluer på testdata
final_fit_jp <- final_wf_jp |> 
  last_fit(split = jp_split, metrics = jp_metrics)

# Se de endelige resultater
collect_metrics(final_fit_jp)

# *******************************************************************************************
#                  Spørgsmål 1 & 2: Individuelle prædiktioner pr. kunde                  ----
# *******************************************************************************************

# Hent sandsynligheder fra testdata
# Ved .pred_1 = sandsynlighed for at churne
# Ved .pred_0  = sandsynlighed for at fortsætte
prædiktioner <- collect_predictions(final_fit_jp) |> 
  select(.row, .pred_class, .pred_1, .pred_0, churned)

# Kunder der sandsynligvis churner
sandsynlige_churnere <- prædiktioner |> 
  filter(.pred_class == "1") |> 
  arrange(desc(.pred_1))

cat("Kunder der sandsynligvis churner\n")
print(sandsynlige_churnere)

# Kunder der sandsynligvis fortsætter 
sandsynlige_fortsættere <- prædiktioner |> 
  filter(.pred_class == "0") |> 
  arrange(desc(.pred_0))

cat("Kunder der sandsynligvis fortsætter\n")
print(sandsynlige_fortsættere)

# *******************************************************************************************
#                Spørgsmål 3: Hvilke "fortsættere" er i risiko for sen churn?            ----
# *******************************************************************************************
# Tager de abonnenter der er prædikteret til at fortsætte,
# men som stadig har en relativt høj churn-sandsynlighed
# Vi bruger 0.3 som grænse
sen_churn_risiko <- sandsynlige_fortsættere |> 
  filter(.pred_1 > 0.3) |> 
  arrange(desc(.pred_1))

cat("Spørgsmål 3: Fortsættere med høj risiko for sen churn\n")
print(sen_churn_risiko)

# Vi laver Confusion Matrix på testdata
collect_predictions(final_fit_jp) |> 
  conf_mat(truth = churned, estimate = .pred_class)

# ROC kurve for den bedste model
collect_predictions(final_fit_jp) |> 
  roc_curve(truth = churned, .pred_1) |> 
  autoplot() +
  ggtitle(paste("ROC kurve -", bedste_jp_id))

# Variable importance for den bedste model
extract_workflow(final_fit_jp) |> 
  extract_fit_parsnip() |> 
  vip::vip(geom = "col")

# *******************************************************************************************
#                    Spørgsmål 4: Hvilke features forklarer churn bedst?                 ----
# *******************************************************************************************
importance <- extract_workflow(final_fit_jp) |> 
  extract_fit_parsnip() |> 
  vi() |> 
  arrange(desc(Importance))

print(importance, n = Inf)

# Top-15 features
importance |> 
  slice_max(Importance, n = 15) |> 
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "#1D9E75", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Top 15 features: Hvad driver churn?",
    x     = "",
    y     = "Importance"
  ) +
  theme_minimal()

# *******************************************************************************************
#                                        Klyngeanalyse                                   ----
# *******************************************************************************************
##                                         K-means                                       ----
# *******************************************************************************************
jp_cluster_vars <- model_data |> 
  select(
    age_at_order,
    account_active_days,
    page_views,
    avg_scroll,
    days_with_activity,
    newsletter_growth,
    previous_subscriptions
  ) |> 
  drop_na()

jp_cluster_scaled <- scale(jp_cluster_vars)

# Kører kmeans 6 gange, med nstart 100 kører vi 100 
# forskellige startpunkter for at finde det optimale
elbow <- map_dbl(1:6, function(k) {
  set.seed(42)
  kmeans(jp_cluster_scaled, centers = k, nstart = 100)$tot.withinss
})

plot(1:6, elbow, 
  type = "b",
  pch = 19,
  xlab = "Antal klynger (k)",
  ylab = "Total within-cluster sum of squares",
  main = "Elbow-metode: optimalt antal klynger")

set.seed(42)
jp.out <- kmeans(jp_cluster_scaled, centers = 3, nstart = 100)

# Tjek klyngerne — størrelse og spredning
jp.out$size
jp.out$tot.withinss
jp.out$withinss

# *******************************************************************************************
##                                       Hierarkisk                                      ----
# *******************************************************************************************
# Hierarkisk clustering med 3 linkage-metoder — som underviserens tilgang
jp_hclust_complete <- hclust(dist(jp_cluster_scaled), method = "complete")
jp_hclust_average  <- hclust(dist(jp_cluster_scaled), method = "average")
jp_hclust_single   <- hclust(dist(jp_cluster_scaled), method = "single")

# Viser alle tre side om side så vi kan sammenligne
par(mfrow = c(1, 3))
plot(jp_hclust_complete, main = "Complete Linkage", xlab = "", sub = "", labels = FALSE, hang = -1)
plot(jp_hclust_average,  main = "Average Linkage",  xlab = "", sub = "", labels = FALSE, hang = -1)
plot(jp_hclust_single,   main = "Single Linkage",   xlab = "", sub = "", labels = FALSE, hang = -1)
par(mfrow = c(1, 1))

# Vi bruger complete til videre analyse, da den typisk giver de mest balancerede klynger
rect.hclust(jp_hclust_complete, k = 3, border = c("#E69F00", "#0072B2", "#009E73"))

# Sammenlign k-means og hierarkisk
table(
  kmeans = jp.out$cluster,
  hclust = cutree(jp_hclust_complete, k = 3)
)

# Kombiner klynge-labels med de originale uskalerede data 
jp_cluster <- data.frame(jp_cluster_vars, klynge = as.factor(jp.out$cluster))

# Laver klyngeprofil
jp_cluster_profile <- jp_cluster |> 
  group_by(klynge) |> 
  summarise(
    age_at_order           = mean(age_at_order),
    account_active_days    = mean(account_active_days),
    page_views             = mean(page_views),
    avg_scroll             = mean(avg_scroll),
    days_with_activity     = mean(days_with_activity),
    newsletter_growth      = mean(newsletter_growth),
    previous_subscriptions = mean(previous_subscriptions)
  )

print(jp_cluster_profile)

# Visualisér klyngeprofiler som grouped barchart 
jp_cluster_profile |> 
  gather("Feature", "Gennemsnit", -klynge) |> 
  ggplot(aes(Feature, Gennemsnit, fill = klynge)) +
  geom_bar(position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Klyngeprofiler — JP abonnenter",
    x     = "",
    fill  = "Klynge"
  )

# Tilføj churnrate per klynge 
model_data_cluster <- model_data |> 
  drop_na(age_at_order, account_active_days, page_views,
          avg_scroll, days_with_activity, 
          newsletter_growth, previous_subscriptions) |> 
  mutate(klynge = as.factor(jp.out$cluster))

model_data_cluster |> 
  group_by(klynge) |> 
  summarise(
    antal      = n(),
    churn_rate = mean(churned == "1") |> round(2)
  )

# *******************************************************************************************
#                    Spørgsmål 5: Kundeprofiler koblet til prædiktioner                  ----
# *******************************************************************************************
# Tilføjer de rækker der bruges i klyngeanalysen
cluster_rows <- model_data |> 
  drop_na(age_at_order, account_active_days, page_views,
          avg_scroll, days_with_activity,
          newsletter_growth, previous_subscriptions)

# Lav prædiktioner KUN på de samme rækker
cluster_prædiktioner <- predict(
  extract_workflow(final_fit_jp),
  new_data = cluster_rows,
  type     = "prob"
)

# Tilføjer klynge-labels
model_data_profil <- cluster_rows |> 
  mutate(
    klynge              = as.factor(jp.out$cluster),
    churn_sandsynlighed = cluster_prædiktioner$.pred_1
  )

# Kundeprofil: én række pr. klynge med de vigtigste nøgletal
kundeprofil <- model_data_profil |> 
  group_by(klynge) |> 
  summarise(
    antal                   = n(),
    churnrate               = mean(churned == "1") |> round(2),
    gns_churn_sandsynlighed = mean(churn_sandsynlighed) |> round(3),
    gns_age_at_order        = mean(age_at_order) |> round(1),
    gns_active_days         = mean(account_active_days) |> round(1),
    gns_page_views          = mean(page_views) |> round(1),
    gns_scroll              = mean(avg_scroll) |> round(2),
    gns_days_with_activity  = mean(days_with_activity) |> round(1),
    .groups = "drop"
  )

print(kundeprofil)

# Churn sandsynlighed pr. klynge som boxplot
ggplot(model_data_profil, aes(x = klynge, y = churn_sandsynlighed, fill = klynge)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    title = "Churn sandsynlighed fordelt på kundesegmenter",
    x     = "Klynge",
    y     = "Churn",
    fill  = "Klynge"
  ) +
  theme_minimal() +
  theme(legend.position = "none")