## Ronald Carter
## Purpose: Paper 4
# ---
# ---

# research inquiry: Exploring the relationship between race, partid, and gunlaw
# data: 2020 sample data from the General Social Survey (GSS)
# install packages
install.packages("tidyverse", repos = "http://cran.us.r-project.org")
install.packages("remotes")
install.packages("tidyverse")
install.packages("tidyr")
install.packages("survey")
install.packages("srvyr")

# load gssr package 
# remotes::install_github("kjhealy/gssr")

# load libraries 
library(gssr)
library(critstats)
library(dplyr)
library(tidyr)
library(ggplot2)
library(haven)
library(tibble)
library(survey)
library(srvyr)
library(forcats)

# laod master file documentation 
data(gss_all)
View(gss_doc)

# closer look into documentation 
data(gss_dict)
gss_dict

# inspect data 
gss_dict %>% 
  filter(variable == "gunlaw")

gss_dict %>% 
  filter(variable == "race")

gss_dict %>% 
  filter(variable == "partyid")


# check years race variable is available
gss_which_years(gss_all, race)


# check years gunlaw variable is available
gss_which_years(gss_all, gunlaw)


# check years partyid variable is available
gss_which_years(gss_all, partyid)

# all available information under one view.
gss_all %>%
  gss_which_years(c(race, partyid, gunlaw)) %>%
  print(n = Inf)


#select rows
gss_all %>% 
  select(race, partyid, gunlaw, year, wtssall)




# convert labeled columns to factors
gss_all$race <- as_factor(gss_all$race)
gss_all$partyid <- as_factor(gss_all$partyid)

# check unique values in partyid column
unique_partyid <- unique(gss_all$partyid)
cat(unique_partyid, sep = "\n")

# retrieve all factor levels in partyid column
factor_levels <- levels(gss_all$partyid)
factor_levels






# keep only responses white black and other for race and recode partyid
gss_all <- gss_all %>%
  filter(race %in% c("white", "black", "other")) %>%
  droplevels()

# check unique values in the race column again
unique_race <- unique(gss_all$race)
print(unique_race)


# count occurrences of each unique value in race column
countsrace <- gss_all %>% count(race)
print(countsrace)


# define all republican and democrat influences 
gss_all$partyid <- fct_collapse(
  gss_all$partyid,
  Democrat = c(
    "strong democrat",
    "not very strong democrat",
    "independent, close to democrat"
  ),
  Republican = c(
    "independent, close to republican",
    "not very strong republican",
    "strong republican"
  )
)

# keep only democrat and republican categories in the partyid column
gss_all <- gss_all %>%
  filter(partyid %in% c("Democrat", "Republican")) %>%
  droplevels()

# check unique values in the partyid column again
unique_partyid <- unique(gss_all$partyid)
print(unique_partyid)

# count occurrences of each unique value in race column
countsparty1 <- gss_all %>% count(partyid)
print(countsparty1)









# check unique values in race column
unique_race <- unique(gss_all$race)
print(unique_race)

# check unique values in party column
unique_partyid <- unique(gss_all$partyid)
print(unique_partyid)




# gunlaw is a factor
gss_all$gunlaw <- as_factor(gss_all$gunlaw)

# create a new column gunlaw_binary (1 for favor, 0 for oppose)
gss_all$gunlaw_binary <- as.integer(gss_all$gunlaw == "favor")


# check unique values in gunlaw_binary column
unique_values <- unique(gss_all$gunlaw_binary)
print(unique_values)

# count occurrences of each unique value in gunlaw_binary column
counts <- gss_all %>% count(gunlaw_binary)
print(counts)


# create dataframe with selected variables for 2016 year 
gss_all %>%
  filter(year == 2016) %>%
  select(year, race, partyid, gunlaw, gunlaw_binary) %>%
  drop_na() -> df16



# create a frequency table and bar graph of race 2016
table.party = table(df16$race)
table.party
barplot(table.party,
        main = "Bar graph of race (2016)",
        xlab = "Respondent Race",
        ylab = "Frequency")

# create a frequency table and bar graph of partyid 2016
table.party = table(df16$partyid)
table.party
barplot(table.party,
        main = "Bar graph of partyid (2016)",
        xlab = "Respondent Partyid",
        ylab = "Frequency")
summary(df16$gunlaw_binary)

# create a frequency table and bar graph of gunlaw 2016
table.party = table(df16$gunlaw_binary)
table.party
barplot(table.party,
        main = "Bar graph of gunlaw_binary (2016)",
        xlab = "Respondent gunlaw",
        ylab = "Frequency")

# do the same thing for 2012 year 
# create dataframe with selected variables for 2012 year
gss_all %>%
  filter(year == 2012) %>%
  select(year, race, partyid, gunlaw, gunlaw_binary) %>%
  drop_na() -> df12




# create a frequency table and bar graph of race 2012
table.party = table(df12$race)
table.party
barplot(table.party,
        main = "Bar graph of race (2012)",
        xlab = "Respondent Race",
        ylab = "Frequency")

# create a frequency table and bar graph of partyid 2012
table.party = table(df12$partyid)
table.party
barplot(table.party,
        main = "Bar graph of partyid (2012)",
        xlab = "Respondent Partyid",
        ylab = "Frequency")
# create a frequency table and bar graph of gunlaw 2012
table.party = table(df12$gunlaw_binary)
table.party
barplot(table.party,
        main = "Bar graph of gunlaw_binary (2012)",
        xlab = "Respondent gunlaw",
        ylab = "Frequency")

# create an interaction term between race and partyid (2016)
df16$race_party_interaction <- interaction(df16$race, df16$partyid)

# create an interaction term between race and partyid (2012)
df12$race_party_interaction <- interaction(df12$race, df12$partyid)

# run logistic regression with the interaction term (2016)
model16 <- glm(gunlaw_binary ~ race + partyid, data = df16, family = "binomial")

# summary of the model
summary(model16)

# run logistic regression with the interaction term (2012)
model12 <- glm(gunlaw_binary ~ race + partyid, data = df12, family = "binomial")

# Summary of the model
summary(model12)


levels(df16$race)
levels(df16$partyid)


#interaction regression
df16$race_party_interaction <- interaction(df16$race, df16$partyid)
model16_interaction <- glm(gunlaw_binary ~ race + partyid + race_party_interaction, 
                           data = df16, family = "binomial")

summary(model16_interaction)


df12$race_party_interaction <- interaction(df12$race, df12$partyid)
model12_interaction <- glm(gunlaw_binary ~ race + partyid + race_party_interaction, 
                           data = df12, family = "binomial")

summary(model12_interaction)