import pandas as pd
import matplotlib.pyplot as plt

googleplaystore = pd.read_csv('googleplaystore.csv')
user_reviews = pd.read_csv('googleplaystore_user_reviews.csv')

# Delete any review that contain neither a Translated_Review nor a Sentiment
user_reviews = user_reviews[user_reviews.Translated_Review.notnull()]
user_reviews = user_reviews[user_reviews.Sentiment.notnull()]
user_reviews.reset_index(drop=True)

# Remove any apps that Rating is invalid (i.e., greater than 5)
googleplaystore = googleplaystore[googleplaystore.Rating <= 5]
googleplaystore = googleplaystore[googleplaystore.Rating.notnull()]
googleplaystore.reset_index(drop=True)

# Produce a pie chart with the Android Ver requirements for the different apps. 
# Group together all versions that make up less than 5% of of the total apps into a single "others" category. 

android_ver = googleplaystore['Android Ver'].value_counts()
android_ver = android_ver[android_ver > 0]

android_ver['others'] = android_ver[android_ver < len(googleplaystore['App'])  * 0.05].sum()
android_ver = android_ver[android_ver >= len(googleplaystore['App']) * 0.05]

android_ver.plot.pie(figsize=(4, 4), title='Android Version')
plt.ylabel('')
plt.show()

# Create a similar pie chart for app Category. 
# In this case, group together categories that make up less than 3% of the apps

app_category = googleplaystore['Category'].value_counts()
app_category = app_category[app_category.notnull()]

app_category['others'] = app_category[app_category < 0.03*len(googleplaystore['App'])].sum()
app_category = app_category[app_category >= 0.03*len(googleplaystore['App'])]

app_category.plot.pie(figsize=(4, 4), title='App Category')
plt.ylabel('')
plt.show()

# Show histograms of the Rating and Review side-by-side across all apps, with 20 bins each.
# Plot the histograms side by side in subplots

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].hist(googleplaystore['Rating'], bins=20, label="Rating")
ax[0].grid(True)
ax[0].set_title("Rating")


reviews = googleplaystore["Reviews"].replace("M", "", regex=True).astype(float)
ax[1].hist(reviews, bins=20, label="Reviews")
ax[1].grid(True)
ax[1].set_title("Reviews")

plt.tight_layout()
plt.show()

# Group the data by 'Sentiment' and count the occurrences
sentiment_counts = user_reviews.groupby('Sentiment').size()

# Create a bar chart
plt.bar(sentiment_counts.index, sentiment_counts.values)

# Set the title and labels
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Display the plot
plt.show()

# Combine the two DataFrames into one, based on the App names. 
# Make sure that all apps from the apps DataFrame are kept, and no app beyond those is added. 
# In other words, the new dataset should have the same amount of unique apps as in the app dataset.

googleplaystore = googleplaystore.drop_duplicates(subset='App')
combined_dataset = pd.merge(googleplaystore, user_reviews, on='App')

# Get a DataFrame of Paid Game apps. Sort the DataFrame by Rating and total numbers of Reviews. 
# drop rows with duplicate app names

paid = googleplaystore[googleplaystore['Type'] == 'Paid']
paid = paid.sort_values(['Rating', 'Reviews'], ascending=[False, False])