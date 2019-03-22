#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
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

# In[2]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[3]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[4]:


# Get just the rows (axis 0)
df.shape[0]


# c. The number of unique users in the dataset.

# In[5]:


#use pandas to return # of unique user ID's
unique_users = df.user_id.nunique()
unique_users


# d. The proportion of users converted.

# In[6]:


# get the number of converted users (converted = true/1)
converted_count = df[df.converted ==1].nunique()
converted_count


# In[7]:


conversion_rate = converted_count/unique_users
conversion_rate
# about 12% conversion rate


# e. The number of times the `new_page` and `treatment` don't match.

# In[8]:


old_control = df.query('landing_page == "new_page" & group != "treatment"')
old_control.count()


# f. Do any of the rows have missing values?

# In[9]:


df.isnull().sum()
# that's a negative, captain


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[10]:


# grab the rows where treatment doesn't equal New Page - we're going to keep those
# we already know which rows are landing on the new page & are in the control group
df2 = df.query('landing_page == "new_page" & group == "treatment" | landing_page == "old_page" & group == "control"')
df2.shape


# In[11]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# In[12]:


# I just want to check my data, my last few tries weren't so successful and I ended up with a bunch of naans
df2.head()


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[13]:


unique_users = df2.user_id.nunique()
unique_users


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[14]:


# check the user ID column for dupes
# here's our culprit
df2[df2.duplicated('user_id')]


# c. What is the row information for the repeat **user_id**? 

# In[15]:


df2[df2.duplicated('user_id', keep=False)]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[16]:


# drop the dupe
df2 = df2.drop_duplicates('user_id')
# make sure it worked
sum(df2.user_id.duplicated())


# In[17]:


# let's see which one made the cut
df2.query('user_id == 773192')


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[18]:


# number of users converted
converted_df = df2[df2.converted ==1]
converted_count = len(converted_df['user_id'])
converted_count


# In[19]:


converted_count/unique_users


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[20]:


# number of users in the control group
control_df = df2[df2.group == "control"]
control_count = len(control_df['user_id'])
control_count


# In[21]:


# number of converted users in the control group
cont_con = control_df[control_df.converted ==1]
cc_count = len(cont_con['user_id'])
cc_count


# In[22]:


cc_count/control_count


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[23]:


# number of users in the treatment group
treat_df = df2[df2.group == "treatment"]
treat_count = len(treat_df['user_id'])
treat_count


# In[24]:


# number of converted users in the treatment group
cont_treat = treat_df[treat_df.converted ==1]
ct_count = len(cont_treat['user_id'])
ct_count


# In[25]:


ct_count/treat_count


# d. What is the probability that an individual received the new page?

# In[26]:


# number of users with new_page as the landing page
new_page = df2[df2.landing_page == "new_page"]
np_count = len(new_page['user_id'])
np_count


# In[27]:


np_conv = np_count/unique_users
np_conv


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **Observe and Report**
# - Overall probability of conversion: 11.96%
# - Control group probability of conversion: 12.03%
# - Treatment group probability of conversion: 11.88%
# - Conversion probability for users who landed on the new page: 50%
# 
# All of the figures are split down the middle.  The number of overall conversions, conversions from the treatment group, and conversions from the control group are all within .25% of each other.  Conversion to non-conversion users landing on the new page ratio is 1:1.  Based on this information, it is safe to assume that landing on the new page did not affect conversion rate.  

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

# **Hypothesis:
# 
# Null - the new page isn't better if the conversion rate is less than or the same as the old page
# 
# H0  :  **$p_{new}$**  ≤  **$p_{old}$**
# 
# Alternative - we can surmise the page is better if the new page conversion rate is higher than the old page
# 
# H1  :  **$p_{new}$**  >  **$p_{old}$**

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[28]:


pnew = df2['converted'].mean()
print(pnew)


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[29]:


pold = df2['converted'].mean()
print(pold)


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[30]:


print(np_count)


# d. What is $n_{old}$, the number of individuals in the control group?

# In[31]:


print(control_count)


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[32]:


new_page_converted = np.random.binomial(1, pnew, np_count)
new_page_converted


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[33]:


old_page_converted = np.random.binomial(1, pold, control_count)
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[34]:


p_difference = pnew-pold


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[35]:


p_diffs = []
for _ in range(10000):
    new_page_converted = np.random.choice([1, 0], size=np_count, p=[pnew, (1-pnew)]).mean()
    old_page_converted = np.random.choice([1, 0], size=control_count, p=[pold, (1-pold)]).mean()
    p_difference = new_page_converted - old_page_converted 
    p_diffs.append(p_difference)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[36]:


plt.hist(p_diffs)
plt.title('10k Simulated Values');


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[37]:


pold = df2[df2.landing_page == 'old_page'].converted.mean()
pnew = df2[df2.landing_page == 'new_page'].converted.mean()
p_difference = pnew-pold
p_difference
(p_diffs > p_difference).mean()


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **We have just computed the p-value (probability value), which is .9073 (roughly 91%).  A p-value is the likelihood of getting the results that we did, given that the conversion rate is the same for both, and we have a null hypothesis.**
# 
# **We also simulated the difference between conversion rates of the old and new landing pages, based on a sample of 10,000 users.**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[38]:


convert_old = cc_count
convert_new = ct_count
n_old = control_count
n_new = np_count


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[39]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [control_count, np_count], alternative='smaller')
z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# In[40]:


from scipy.stats import norm

# c-value @ 95% confidence
norm.ppf(1-(0.05/2))


# **The z-score is less than c-value, therefore we cannot reject the null hypothesis.  The p-value yielded results similar to our manual computation, which indicates that the numbers agree, we should not reject the null hypothesis.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic regression.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[41]:


df2.head()


# In[42]:


df2['intercept']=1
df2.head()


# In[43]:


df2['ab_page'] = np.where(df2['landing_page'] == 'new_page', 1, 0)
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[57]:


logit = sm.Logit(df2['converted'],df2[['intercept','ab_page']])


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[58]:


results = logit.fit()
results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **The p-value is .19, which will be different from the results of Part II, because we didn't apply as many parameters.  We're only measuring whether the user was served the old page or the new page, and not taking conversion into account.**
# 
# 

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **It is always a good idea to consider other factors, like context, time spent in certain areas of the site, demographic, and persona.  This will help you discover the customer journey and tell you who the end-user really is.  You would then be able to iterate over the current implementation to be more appealing to your user base.  A possible disadvantage would be having a very bulky dataset**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[59]:


# let's read it in and take a quick look
countries_df = pd.read_csv('./countries.csv')
countries_df.head()


# In[60]:


# ok now let's join the tables
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.head()


# In[48]:


# See how many different countries our user base is coming from.  
# This will help determine how many columns to add and what to label them.
df_new['country'].value_counts()


# In[62]:


# it makes more sense to add 3 columns instead of just 2.  
# bundling CA & US into a 'north america' column isn't granular enough
# add the dummy variables to the dataframe
df_new[['CA', 'US', 'UK']] = pd.get_dummies(df_new['country'])[['CA','US','UK']]
df_new.head()


# In[73]:


logit_u = sm.Logit(df_new['converted'], df_new[['intercept', 'UK','US']])
results = logit_u.fit()
results.summary()


# In[74]:


np.exp(results.params)


# In[75]:


logit_c = sm.Logit(df_new['converted'], df_new[['intercept', 'CA','US']])
results = logit_c.fit()
results.summary()


# In[76]:


np.exp(results.params)


# **The numbers do not show significant difference in conversion rates across the 3 countries.  All of the p-values are higher than the error rate of .05%, which also indicates no statistical significance.  UK was slightly more likely to convert than the US, US was about 4% more likely to convert than CA.**

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[77]:


df_new['CA_ab'] = df_new['CA']*df_new['ab_page']
df_new['UK_ab'] = df_new['UK']*df_new['ab_page']
df_new['US_ab'] = df_new['US']*df_new['ab_page']
df_new.head()


# In[81]:


logit_cab = sm.Logit(df_new['converted'], df_new[['intercept', 'ab_page', 'UK', 'US', 'UK_ab', 'US_ab']])
results = logit_cab.fit()
results.summary()


# In[79]:


np.exp(results.params)


# In[82]:


logit_cab2 = sm.Logit(df_new['converted'], df_new[['intercept', 'ab_page', 'CA', 'US', 'CA_ab', 'US_ab']])
results = logit_cab2.fit()
results.summary()


# In[83]:


np.exp(results.params)


# **The p-values are still higher than the error rate of .05%, which supports the conclusion that there is no statistical significance, and neither the user's landing page nor country of origin had any bearing on conversion rate.**

# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

