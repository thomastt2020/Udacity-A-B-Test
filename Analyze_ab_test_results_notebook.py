
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[5]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[6]:


# import data
df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[7]:


# use describe or shape to find number of rows
df.describe()


# In[8]:


df.shape


# c. The number of unique users in the dataset.

# In[9]:


df.nunique()


# d. The proportion of users converted.

# In[10]:


df.converted.mean()


# e. The number of times the `new_page` and `treatment` don't line up.

# In[11]:


df.query ('group == "treatment" and landing_page != "new_page"')


# In[12]:


df.query ('group == "control" and landing_page != "old_page"')


# f. Do any of the rows have missing values?

# In[13]:


# check if any null values
df.info()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[14]:


# put all treatment AND new_page into one dataframe
df2t = df.query('group == "treatment" and landing_page == "new_page"')


# In[15]:


# put all control AND old_page into one dataframe
df2c = df.query('group == "control" and landing_page == "old_page"')


# In[16]:


# merge two properly aligned dataframes together
df2 = df2t.merge(df2c, how='outer')


# In[17]:


df2.shape


# In[18]:


df2.describe()


# In[19]:


df2.head()


# In[20]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[21]:


df2.nunique()


# 290584 user_id         290585 timestamp       

# b. There is one **user_id** repeated in **df2**.  What is it?

# In[22]:


sum(df2.user_id.duplicated())


# c. What is the row information for the repeat **user_id**? 

# In[23]:


df2[df2.duplicated(['user_id'], keep=False)]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[24]:


df2 = df2[~df2.user_id.duplicated(keep='first')]
# https://stackoverflow.com/questions/13035764/remove-rows-with-duplicate-indices-pandas-dataframe-and-timeseries


# In[25]:


df2.shape


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[26]:


df2.converted.mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[27]:


df2_control = df2.query('group == "control"')
df2_control.converted.mean()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[28]:


df2_treatment = df2.query('group == "treatment"')
df2_treatment.converted.mean()


# d. What is the probability that an individual received the new page?

# In[29]:


len(df2_treatment.index)/len(df2.index)


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# **ANSWER:**
# It appears that individuals in the treatment group had a conversion rate of 11.88% and individuals in the control grounp had a conversion rate of 12.04%. This leads us to think that the treatment group does not lead to more conversions than the treatment group. However, it remains to be seen if this is true, or due to some bias.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **ANSWER:** If we assume the old page is better unless the new page proves to be definately better, then the null hypotheses is that the mean converted rate of the old page is greater or equal to the converted rate of the new page and the alternative hypothesis is that the mean converted rate of the new page is greater than the converted rate of the old page.
# 
# 
# Null Hypotheses:  **$p_{old}$** is equal greater or equal to  **$p_{new}$**
# 
# Alternative Hypothesis:  **$p_{old}$** is less than **$p_{new}$**

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# **$p_{new}$** = df2.converted.mean() = 0.11959708724499628

# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# **$p_{old}$** = df2.converted.mean() = 0.11959708724499628

# c. What is $n_{new}$?

# In[30]:


n_new = len(df2_treatment.index)
n_new


# d. What is $n_{old}$?

# In[31]:


n_old = len(df2_control.index)
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[32]:


# simulate under null for n_new
new_page_converted = np.random.choice([1, 0], size=len(df2_treatment.index), p=[df2.converted.mean(), (1-(df2.converted.mean()))])


# In[33]:


plt.hist(new_page_converted);


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[34]:


# simulate under null for n_new
old_page_converted = np.random.choice([1, 0], size=len(df2_control.index), p=[df2.converted.mean(), (1-(df2.converted.mean()))])


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[35]:


new_page_converted.mean() - old_page_converted.mean()


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[37]:


# Create a sampling distribution of the difference in proportions
#p_diffs = []
#for _ in range(10000):
#    new_page_converted = np.random.choice([1, 0], size=len(df2_treatment.index), p=[df2.converted.mean(), (1-(df2.converted.mean()))])
#    old_page_converted = np.random.choice([1, 0], size=len(df2_control.index), p=[df2.converted.mean(), (1-(df2.converted.mean()))])
#    p_diffs.append(new_page_converted.mean() - old_page_converted.mean())


# **A faster way to simulate the 10000 trials** Require Changes
# 
# When possible, it is always more computationally efficient to use numpy built-in operations over explicit for loops. The short reason is that numpy-based operations attack a computational problem based on vectors by computing large chunks simultaneously.
# 
# Additionally, using loops to simulate 10000 can take a considerable amount of time vs using numpy
# https://softwareengineering.stackexchange.com/questions/254475/how-do-i-move-away-from-the-for-loop-school-of-thought
# 
# new_converted_simulation = np.random.binomial(n_new, p_new,  10000)/n_new
# old_converted_simulation = np.random.binomial(n_old, p_old,  10000)/n_old
# p_diffs = new_converted_simulation - old_converted_simulation
# 
# Essentially, we are applying the null proportion to the total size of each page using the binomial distribution. Each element, for example, innp.random.binomial(n_new, p_new, 10000) results in an array with values like [17262, 17250, 17277...]. This array is 10000 elements large
# 
# When we divide it by n_new, Python broadcasts n_new for each element and we return a proportion for each element.
# 
# This is essentially is simulating, 10000, the new page conversion rate.
# 
# We do this again for the old page.
# 
# The difference of the two will result in a simulated difference array of length 10000 between the new page and old page conversions.
# 
# Note that this method does not require you to calculate the null values to get the p-value.

# In[38]:


new_page_converted = np.random.binomial(n_new, df2.converted.mean(),  10000)/n_new

old_page_converted = np.random.binomial(n_old, df2.converted.mean(),  10000)/n_old

p_diffs = new_page_converted - old_page_converted


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# 1. We took the entire dataset, 
# 2. removed rows with issues (mismatching or duplicates)
# 3. Used the mean for the entire dataset
# 4. Applied to number of new_pages and number of old_pages
# 5. then took the difference
# 6. the difference, based on large numbers, and central theory, should be around 0 because the mean was the same for the new_pages and number of old_pages

# In[39]:


# calculate mean of p_diffs
np.array(p_diffs).mean()


# In[40]:


ab_data_diff = df2_treatment.converted.mean() - df2_control.converted.mean()
ab_data_diff


# In[41]:


(p_diffs > ab_data_diff).mean()


# In[42]:


(p_diffs < ab_data_diff).mean()


# In[43]:


# The parameter is less than some value in the alternative hypothesis

# low is the difference between treatment and control groups in ab_data
low = ab_data_diff

# low is the difference between treatment and control groups in null hypothesis
high = (np.array(p_diffs).mean())

plt.hist(p_diffs);
plt.title('Simulation under null hypothesis')
plt.xlabel('Difference in mean between treatment and control groups')
plt.ylabel('Frequency')
plt.axvline(x=high, color='r', linewidth=2, label='mean of null difference')
plt.axvline(x=low, color='b', linewidth=2, label='mean of ab_data difference')
plt.legend()


# In[44]:


(np.array(p_diffs).mean()) - ab_data_diff


# j. What proportion of the p_diffs are greater than the actual difference observed in ab_data.csv

# In[45]:


# compute p-value
(p_diffs > ab_data_diff).mean()


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **ANSWER:**  The proportion of the p_diffs that are greater than the actual difference observed in ab_data.csv is called the **p-value**
# 
# A p-value is the probability of observing your statistic if the null hypothesis is true.
# 
# The null hypothesis was that the difference in means would be equal or less than 0, and the alternative was the difference would be greater than 0. However, the difference is less than zero, and the p-value is very large. We do not have evidence to rejust the null.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[46]:


import statsmodels.api as sm

convert_old = len(df2_control[df2_control['converted'] == 1])
convert_new = len(df2_treatment[df2_treatment['converted'] == 1])
n_old = len(df2_control.index)
n_new = len(df2_treatment.index)


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[47]:


#z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new])
#z_score, p_value


# **Missing Z-test Parameters** Requires Changes
# 
# Please note that the pvalue here should correspond to the the answer in J, which should be ~0.9.
# 
# Please refer to the Z-proportions test documentation.
# 
# Also, refer to the Part II instructions where we are assuming a directional set of hypotheses statements
# 
# As you can see, we require that the alternative hypothesis to assume the new_page is better than the old_page. The default parameter for the statsmodels.stats.proportion.proportions_ztest is “two-sided” and it will assume the alternative is assuming simply an inequality (new_page!=old_page) rather than new is greater than the old (new_page > old_page). 
# 
# Please refer to the documentation on how you modify the function in order to achieve the the alternative hypothesis of new>old. Hint You are looking for the alternative parameter.
# 
# http://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportions_ztest.html

# In[49]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **ANSWER:**  In statistics, the standard score is the signed number of standard deviations by which the value of an observation or data point is above the mean value of what is being observed or measured (wikipedia)
# 
# Before this test began, we would have picked a significance level. Let's just say it's 95%. Since this is a test for the difference, it's a two-tail test so a z-score past -1.96 or 1.96 will be significant. (knowledgetack)
# 
# The conversion rate of the new landing page is only 1.3109 standard deviations from the conversion rate of the old landing page. This is less than the critical value of 1.96. We cannot reject the hull hypothesis that the difference between the two conversion rates is no different from zero.

# Also, the p-value is 0.9050. The p-value was calculated where the null hypothesis was that the new page would convert more than the old page, and the alternative was the old page converted more than or equal to the new page.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **ANSWER:** In statistics, linear regression is a linear approach for modelling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X. The case of one explanatory variable is called simple linear regression. (wikipedia)
# 
# There is only one explanatory variable in this case.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[50]:


df2['intercept']=1


# In[51]:


df2['ab_page']=0


# In[52]:


ab_page_index = df2[df2['group']=='treatment'].index


# In[53]:


df2.loc[ab_page_index, "ab_page"] = 1


# In[54]:


df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[55]:


lm = sm.OLS(df2['converted'], df2[['intercept', 'ab_page']])
results=lm.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[56]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **ANSWER:** The p-value associated with ab_page is 0.190
# 
# In Part II, the p-value was calculated where the null hypothesis was that the new page would convert more than the old page, and the alternative was the old page converted more than or equal to the new page.
# 
# In Part III, we used variables, and used a linear model to determine the p-value. The null hypothesis was that the difference between the pages is equal to 0, and the alternative hypothesis was the difference between the pages is greater or less than 0.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **ANSWER:** It would be a good idea to consider other factors to add into the regression model. Perhaps time of day that the user used the page, might influence when people sign up online. The disadvantage is it adds complexity, because variables may affect other variables. Also, some variables may not affect the outcome.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[57]:


df_countries = pd.read_csv('countries.csv')
df_countries.head()


# In[58]:


df_countries.country.unique()


# In[59]:


country_dummies = pd.get_dummies(df_countries['country'])
df_new = df_countries.join(country_dummies)


# In[60]:


df_new.head()


# In[61]:


df3 = df2.set_index('user_id').join(df_new.set_index('user_id'))


# In[62]:


lm = sm.OLS(df3['converted'], df3[['intercept', 'UK', 'US']])
results = lm.fit()
results.summary()


# Set Canada as baseline country. The correlation coefficient is very small for both UK and US. This means the relationship between country and conversion is weak.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[63]:


lm = sm.OLS(df3['converted'], df3[['intercept', 'ab_page', 'UK', 'US']])
results = lm.fit()
results.summary()


# The p-value for ab_page is 0.191. 
# 
# The null hypothesis was that the difference in means would be 0, and the alternative was the difference would be greater or less than 0. The p-value is still large. We fail to reject the null hypothesis.

# <a id='conclusions'></a>
# ## Conclusions
# 
# Congratulations on completing the project! 
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by continuing on to the next module in the program.
