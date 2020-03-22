# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:25:59 2020

@author: sukh
"""
#reset
# Import pandas into the environment
import pandas as pd
import numpy as np

# Import marketing.csv 
marketing = pd.read_csv('marketing.csv')


#Examining the data
# Print the first five rows of the DataFrame
print(marketing.head(5))

# Print the statistics of all columns
print(marketing.describe())

# Check column data types and non-missing values
print(marketing.info())

# Change the data type of a column
marketing['converted'] = marketing['converted'].astype('bool')
print(marketing['converted'].dtype)

marketing['is_retained'] = marketing['is_retained'].astype('bool')
print(marketing['is_retained'].dtype)

#Creating new boolean columns
marketing['is_house_ads'] = np.where(
marketing['marketing_channel'] == 'House Ads',
True,
False
)


# Add the new column is_correct_lang
marketing['is_correct_lang'] = np.where(
    marketing.language_displayed==marketing.language_preferred,
    'Yes', 
    'No')


#Mapping values to existing columns
channel_dict = {"House Ads": 1, "Instagram": 2,
"Facebook": 3, "Email": 4, "Push": 5}
#marketing['channel_code'] = marketing['marketing_channel'].map(channel_dict)
marketing['channel_code'] = marketing['subscribing_channel'].map(channel_dict)
print(marketing['channel_code'].head(3))



# Convert already existing column to datetime column
marketing['date_served'] = pd.to_datetime(
    marketing['date_served']
)
marketing['date_subscribed'] = pd.to_datetime(
    marketing['date_subscribed']
)
marketing['date_canceled'] = pd.to_datetime(
    marketing['date_canceled']
)

print(marketing.dtypes)

marketing=marketing.drop('day_served', axis=1)
marketing['DoW'] = marketing['date_subscribed'].dt.dayofweek
#Initial exploratory analysis
# Aggregate unique users that see ads by date
daily_users = marketing.groupby(['date_served'])['user_id'].nunique()
print(daily_users)


import matplotlib.pyplot as plt
# Plot
daily_users.plot()
# Annotate
plt.title('Daily number of users who see ads')
plt.xlabel('Date')
plt.ylabel('Number of users')
plt.xticks(rotation = 45)
plt.show()


#Calculating conversion rate using pandas
subscribers = marketing[marketing['converted'] == True]['user_id'].nunique()
total = marketing['user_id'].nunique()
conv_rate = subscribers/total
print(round(conv_rate*100, 2), '%')


# Calculate the number of subscribers
total_subscribers = marketing[marketing.converted==True].user_id.nunique()

# Calculate the number of people who remained subscribed
retained = marketing[marketing.is_retained==True].user_id.nunique()

# Calculate the retention rate
retention_rate = retained/total_subscribers
print(round(retention_rate*100, 2), "%")



#Segmenting using pandas - groupby()



# Group by subscribing_channel and calculate retention
retained = marketing[marketing['is_retained'] == True]\
          .groupby(['subscribing_channel'])\
          ['user_id'].nunique()
print(retained)

# Group by subscribing_channel and calculate subscribers
subscribers = marketing[marketing['converted'] == True]\
              .groupby(['subscribing_channel'])\
              ['user_id'].nunique()
print(subscribers)


# Calculate the retention rate across the DataFrame
channel_retention_rate = (retained/subscribers)*100
print(channel_retention_rate)


#Comparing language conversion rate (I)

# Isolate english speakers
english_speakers = marketing[marketing['language_displayed'] == 'English']

# Calculate the total number of English speaking users
total = english_speakers.user_id.nunique()

# Calculate the number of English speakers who converted
subscribers = english_speakers[english_speakers.converted==True].user_id.nunique()

# Calculate conversion rate
conversion_rate = subscribers/total
print('English speaker conversion rate:', round(conversion_rate*100,2), '%')




# Group by subscribing_channel and calculate subscribers
subscribers_lang = marketing[marketing['converted'] == True]\
              .groupby(['language_displayed'])\
              ['user_id'].nunique()
print(subscribers_lang)

total_lang = marketing.groupby(['language_displayed'])\
              ['user_id'].nunique()
print(total_lang)

language_conversion_rate = subscribers_lang/total_lang
import matplotlib.pyplot as plt
# Create a bar chart using channel retention DataFrame
language_conversion_rate.plot(kind = 'bar')
# Add a title and x and y-axis labels
plt.title('Conversion rate by language\n', size = 16)
plt.xlabel('Language', size = 14)
plt.ylabel('Conversion rate (%)', size = 14)
# Display the plot
plt.show()

#Aggregating by date

# Group by date_served and count unique users
total = marketing.groupby(['date_served'])['user_id'].nunique()

# Group by date_served and count unique converted users
subscribers = marketing[marketing.converted==True].groupby(['date_served'])['user_id'].nunique()

# Calculate the conversion rate per day
daily_conversion_rate = subscribers/total
print(daily_conversion_rate)

# Reset index to turn the results into a DataFrame
daily_conversion_rate = pd.DataFrame(daily_conversion_rate.reset_index(0))

# Rename columns
daily_conversion_rate.columns = ['date_subscribed', 
                              'conversion_rate']


# Create a line chart using daily_conversion_rate
daily_conversion_rate.plot('date_subscribed', 
                              'conversion_rate')

plt.title('Daily conversion rate\n', size = 16)
plt.ylabel('Conversion rate (%)', size = 14)
plt.xlabel('Date', size = 14)

# Set the y-axis to begin at 0
plt.ylim(0)

# Display the plot
plt.show()

#Grouping by multiple columns
language = marketing.groupby(['date_served',
'language_preferred'])\
['user_id'].count()
print(language.head())

#Unstacking after groupby
language = pd.DataFrame(language.unstack(level=1))
print(language.head())

language.plot()
plt.title('Daily language preferences')
plt.xlabel('Date')
plt.ylabel('Users')
plt.legend(loc = 'upper right',
labels = language.columns.values)
plt.show()

# Create DataFrame grouped by age and language preference
language_age = marketing.groupby(['language_preferred',
'age_group'])\
['user_id'].count()

language_age = pd.DataFrame(language_age.unstack(level=1))
print(language_age.head())

language_age.plot(kind='bar')
plt.title('Language preferences by age group')
plt.xlabel('Language')
plt.ylabel('Users')
plt.legend(loc = 'upper right',
labels = language_age.columns.values)
plt.show()


channel_age = marketing.groupby(['marketing_channel', 'age_group'])\
                                ['user_id'].count()

# Unstack channel_age and transform it into a DataFrame
channel_age_df = pd.DataFrame(channel_age.unstack(level=1))

# Plot channel_age
channel_age_df.plot(kind='bar')
plt.title('Marketing channels by age group')
plt.xlabel('Age Group')
plt.ylabel('Users')
# Add a legend to the plot
plt.legend(loc = 'upper right',
labels = channel_age_df.columns.values)
plt.show()




# Divide retained subscribers by total subscribers
retention_rate = retention_subs/retention_total
retention_rate_df = pd.DataFrame(retention_rate.unstack(level=1))

# Plot retention rate
retention_rate_df.plot()

# Add a title, x-label, y-label, legend and display the plot
plt.title('Retention Rate by Subscribing Channel')
plt.xlabel('Date Subscribed')
plt.ylabel('Retention Rate (%)')
plt.legend(loc = 'upper right',
labels = retention_rate_df.columns.values)
plt.show()


def conversion_rate(dataframe, column_names):
    # Total number of converted users
    column_conv = dataframe[dataframe['converted']==True].groupby(column_names).user_id.nunique()

    # Total number users
    column_total = dataframe.groupby(column_names).user_id.nunique()
    
    # Conversion rate 
    conversion_rate = column_conv/column_total
    
    # Fill missing values with 0
    conversion_rate = conversion_rate.fillna(0)
    return conversion_rate


# Calculate conversion rate by age_group
age_group_conv = conversion_rate(marketing, ['date_served', 'age_group'])
print(age_group_conv)

# Unstack and create a DataFrame
age_group_df = pd.DataFrame(age_group_conv.unstack(level=1))

# Visualize conversion by age_group
age_group_df.plot()
plt.title('Conversion rate by age group\n', size = 16)
plt.ylabel('Conversion rate', size = 14)
plt.xlabel('Age group', size = 14)
plt.show()

def plotting_conv(dataframe):
    for column in dataframe:
        # Plot column by dataframe's index
        plt.plot(dataframe.index, dataframe[column])
        plt.title('Daily ' + str(column) + ' conversion rate\n', 
                  size = 16)
        plt.ylabel('Conversion rate', size = 14)
        plt.xlabel('Date', size = 14)
        plt.xticks(rotation = 45)
        # Show plot
        plt.show()  
        plt.clf()



# Calculate conversion rate by date served and age group
age_group_conv = conversion_rate(marketing, ['date_served', 'age_group'])

# Unstack age_group_conv and create a DataFrame
age_group_df = pd.DataFrame(age_group_conv.unstack(level=1))

# Plot the results
plotting_conv(age_group_df)

def retention_rate(dataframe, column_names):
# Group by column_names and calculate retention
    retained = dataframe[dataframe['is_retained'] == True]\
.   groupby(column_names)['user_id'].nunique()
# Group by column_names and calculate conversion
    converted = dataframe[dataframe['converted'] == True]\
.   groupby(column_names)['user_id'].nunique()
    retention_rate = retained/converted
    return retention_rate

daily_retention = retention_rate(marketing,
['date_subscribed',
'subscribing_channel'])
daily_retention = pd.DataFrame(
daily_retention.unstack(level=1)
) 
print(daily_retention.head())



daily_retention.plot()
plt.title('Daily channel retention rate\n', size = 16)
plt.ylabel('Retention rate (%)', size = 14)
plt.xlabel('Date', size = 14)
plt.show()

DoW_retention = retention_rate(marketing, ['DoW'])

# Plot retention by day of week
DoW_retention.plot()
plt.title('Retention rate by day of week')
plt.ylim(0)
plt.show()


# Calculate conversion rate by date served and channel
daily_conv_channel = conversion_rate(marketing, ['date_served', 'marketing_channel'])

print(daily_conv_channel.head())

# Calculate conversion rate by date served and channel
daily_conv_channel = conversion_rate(marketing, ['date_served', 
                                                 'marketing_channel'])

# Unstack daily_conv_channel and convert it to a DataFrame
daily_conv_channel = pd.DataFrame(daily_conv_channel.unstack(level = 1))

# Plot results of daily_conv_channel
plotting_conv(daily_conv_channel)



# Add day of week column to marketing
marketing['DoW_served'] = marketing['date_served'].dt.dayofweek

# Calculate conversion rate by day of week
DoW_conversion = conversion_rate(marketing, [ 'DoW_served','marketing_channel'])


# Unstack channels
DoW_df = pd.DataFrame(DoW_conversion.unstack(level=1))

# Plot conversion rate by day of week
DoW_df.plot()
plt.title('Conversion rate by day of week\n')
plt.ylim(0)

#House ads conversion by language

# Isolate the rows where marketing channel is House Ads
house_ads =marketing[marketing['marketing_channel']=='House Ads']

# Calculate conversion by date served, and language displayed
conv_lang_channel = conversion_rate(house_ads, ['date_served', 'language_displayed'])

# Unstack conv_lang_channel
conv_lang_df = pd.DataFrame(conv_lang_channel.unstack(level=1))

# Use your plotting function to display results
plotting_conv(conv_lang_df)

#Creating a DataFrame for house ads

# Add the new column is_correct_lang
house_ads['is_correct_lang'] = np.where(
    house_ads['language_displayed'] == house_ads['language_preferred'], 
    'Yes', 
    'No')

# Groupby date_served and correct_language
language_check = house_ads.groupby(['date_served','is_correct_lang'])['is_correct_lang'].count()

# Unstack language_check and fill missing values with 0's
language_check_df = pd.DataFrame(language_check.unstack(level=1)).fillna(0)

# Print results
print(language_check_df)

# Divide the count where language is correct by the row sum
# Divide the count where language is correct by the row sum
language_check_df['pct'] = language_check_df['Yes']/language_check_df.sum(axis=1)

# Plot and show your results
plt.plot(language_check_df.index.values, language_check_df['pct'])
plt.show()


# Calculate pre-error conversion rate
house_ads_no_bug = house_ads[house_ads['date_served'] < '2018-01-11']
lang_conv = conversion_rate(house_ads_no_bug,
['language_displayed'])

# Index other language conversion rate against English
spanish_index = lang_conv['Spanish']/lang_conv['English']
arabic_index = lang_conv['Arabic']/lang_conv['English']
german_index = lang_conv['German']/lang_conv['English']

print("Spanish index:", spanish_index)
print("Arabic index:", arabic_index)
print("German index:", german_index)

#Analyzing user preferences
#To understand the true impact of the bug, it is crucial to determine how many subscribers we would have expected had there been no language error.

# Group house_ads by date and language
converted = house_ads.groupby(['date_served', \
'language_preferred'])\
.agg({'user_id':'nunique',\
'converted':'sum'})

# Unstack converted
converted = pd.DataFrame(converted.unstack(level=1))

# Create English conversion rate column for affected period
converted['english_conv_rate'] = converted.loc['2018-01-11':'2018-01-31'][('converted','English')]

# Create expected conversion rates for each language
converted['expected_spanish_rate'] = converted['english_conv_rate']*spanish_index
converted['expected_arabic_rate'] = converted['english_conv_rate']*arabic_index
converted['expected_german_rate'] = converted['english_conv_rate']*german_index

# Multiply number of users by the expected conversion rate
converted['expected_spanish_conv'] = converted['expected_spanish_rate']/100*converted[('user_id','Spanish')]
converted['expected_arabic_conv'] = converted['expected_arabic_rate']/100*converted[('user_id','Arabic')]
converted['expected_german_conv'] = converted['expected_german_rate']/100*converted[('user_id','German')]



# Use .loc to slice only the relevant dates
converted = converted.loc['2018-01-11':'2018-01-31']

# Sum expected subscribers for each language
expected_subs = converted['expected_spanish_conv'].sum() + converted['expected_arabic_conv'].sum() + converted['expected_german_conv'].sum()

# Calculate how many subscribers we actually got
actual_subs = converted[('converted','Spanish')].sum() + converted[('converted','Arabic')].sum() + converted[('converted','German')].sum()

# Subtract how many subscribers we got despite the bug
lost_subs = expected_subs - actual_subs
print(lost_subs)


email = marketing[marketing['marketing_channel'] == 'Email']
allocation = email.groupby(['variant'])['user_id'].nunique()
allocation.plot(kind='bar')
plt.title('Personalization test allocation')
plt.xticks(rotation = 0)
plt.ylabel('# participants')
plt.show()


subscribers = email.groupby(['user_id',
'variant'])['converted'].max()
subscribers = pd.DataFrame(subscribers.unstack(level=1))

# Drop missing values from the control column
control = subscribers['control'].dropna()
# Drop missing values from the personalization column
personalization = subscribers['personalization'].dropna()

#Calculating lift
# Calcuate the mean of a and b
a_mean = np.mean(control)
b_mean = np.mean(personalization)
# Calculate the lift using a_mean and b_mean
lift = (b_mean-a_mean)/a_mean
print("lift:", str(round(lift*100, 2)) + '%')
from scipy.stats import ttest_ind
t = ttest_ind(control, personalization)
print(t)