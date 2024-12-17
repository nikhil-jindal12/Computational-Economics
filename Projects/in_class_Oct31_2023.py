# Task 1: install pandas library.
# if you are using the Anaconda environment, pandas is included by defaul.
# if you are using your own virtual environment, follow this instruction
# https://pandas.pydata.org/docs/getting_started/install.html to install pandas

# Task 2: import the pandas library use "pd" as the alias
import pandas as pd

# ============================================================================
# Section 1. Getting to know Pandas' Data Structures
# ============================================================================

#-----------------------------------------------------------------------------
# 1.1. Understanding Series Objects
#----------------------------------------------------------------------------
# let's create a Pandas series known as "points"
points = pd.Series([2062, 2022, 2004, 2001,1957])
print(points)

# Task 3: check the values in the series "points"
points

# Task 4: check the index in the series "points"
points

# Task 5: use the type() function to check what is the data type of the values
type(points.values)


# Series can also have an arbitrary type of index.
# You can think of this explicit index as labels for a specific row:
club_points = pd.Series(
    [2062, 2022, 2004, 2001, 1957],
    index=["RMD", "MNC", "NAP", "MUN", "LIV"]
)

# Here’s how to construct a Series with a label index from a Python dictionary:
club_fans_count = pd.Series({"RMD": 252.1, "LIV": 92.7})
print(club_fans_count)


# Task 6: call the .keys() method in the club_fans_count, what's the return?


# Task 7: find out is the index "LIV" in the Series club_fans_count?


# Task 8: find out is the index "MNC" in the Series club_fans_count?


# ---------------------------------------------------------------------------
# 1.2. Understanding DataFrame Objects
# ---------------------------------------------------------------------------
# If you’ve followed along with the Series examples,
# then you should already have two Series objects with cities as keys:
print(club_points)
print(club_fans_count)

# Task 9: creaet a dataframe known as club_data by combining
# the Series "club_points" and the Series "club_fans_count"
# Each Series is a column in your DataFrame, the name of the columns are
# "points" and "fans_count"


print(club_data)

# Task 10: access the index attribute of the dataframe club_data



# Task 11: access the shape attribute of the DataFrame club_data



# Task 12: access the axes attribute of the DataFrame club_data



# Task 13: figure out what are the marks for row index and column index
club_data.axes[0]
club_data.axes[1]


# Task 14: call the .keys() method within the club_data, see what the return is


# Task 15, try use the "in" keyword to see if "LIV" is in the club_data DF


# Task 16, try use the "in" keyword to see if "points" is in the club_data DF


# ============================================================================
# Section 2. Accessing Series Elements
# ============================================================================

# ---------------------------------------------------------------------------
# 2.1. Using the Indexing Operator
# ---------------------------------------------------------------------------

# Task 17: use the positional index (i.e., impliit index) to get all elements except the first one in club_points



# Task 18: use the label index (i.e., explicit index) to get all elements after "MNC" in club_points


# The indexing operator [] is convenient, but there’s a caveat. What if the labels are also numbers? Say you have to work with a Series object like this:


# !CAVEAT! numerical label index could be tricky.
# Consider the following Series
colors = pd.Series(
    ["red", "purple", "blue", "green", "yellow"],
    index=[1, 2, 3, 5, 8]
)
print(colors)

# Task 19: What will colors[1] return? Is it the same as you expected?


# ---------------------------------------------------------------------------
# 2.2. Using .loc and .iloc
# ---------------------------------------------------------------------------
# in short:
# .loc refers to the label index.
# .iloc refers to the positional index.
# These data access methods are much more readable:

# Task 20: get the second and third elements from the colors Series using .iloc


# Task 21. pass a negative positional index to .iloc, and see wha the return is


# ============================================================================
# Section 3. Accessing DataFrame Elements
# ============================================================================
"""
Since a DataFrame consists of Series objects, you can use the very same tools
to access its elements. The crucial difference is the additional dimension of
the DataFrame. You’ll use the indexing operator for the columns and the access
methods .loc and .iloc on the rows.
"""
# ---------------------------------------------------------------------------
# 3.1. Using the index operator
# ---------------------------------------------------------------------------

# Task 22: use the index operator [ ] to get the information of the column "points" from the dataframe club_data


# Task 23: try use the dot syntax to get the points information


# ---------------------------------------------------------------------------
# 3.2. Using .loc and .iloc
# ---------------------------------------------------------------------------

# Task 24: use the .loc to get information for the club "LIV"


# task use the .loc to get information from the club "LIV" to "NAP"


# Task 25: use the .iloc to get the information of the second column in the df club_data


# Task 26, get the points information for clubs from "LIV" to "NAP"


# ============================================================================
# Section 4. Combining datasets
# ============================================================================
"""
Another aspect of real-world data is that it often comes in multiple pieces. In this section, you’ll learn how to grab those pieces and combine them into one dataset that’s ready for analysis.
"""
# Earlier, we created a dataset with football club information.

club_points = pd.Series(
    [2062, 2022, 2004, 2001, 1957],
    index=["RMD", "MNC", "NAP", "MUN", "LIV"]
    )
club_fans_count = pd.Series({"RMD": 252.1, "LIV": 92.7})

club_data = pd.DataFrame({
    "points": club_points,
    "fans_count": club_fans_count
    })

# Now, say you've managed to gather more data on two more clubs:

more_club_data = pd.DataFrame(
    {"points": [1908, 1907],
     "fans_count": [250.3, 91.0]},
     index=["FCB", "PSG"]
)


# ---------------------------------------------------------------------------
# 4.1. Using the concate() method to join dataframes
# ---------------------------------------------------------------------------

# Task 27: use .concat() to join club_data and more_club_data together, do not sort the data after joining.


# Task 28: use .concat() to add the countries df as new column of the df all_club_data, name the new df as clubs
countries = pd.DataFrame({
    "country": ["England", "Spain", "Italy", "France"]},
    index=["LIV", "FCB", "NAP", "PSG"]
    )

print(clubs)

# ---------------------------------------------------------------------------
# 4.2. Using the merge() method to join dataframes
# ---------------------------------------------------------------------------

# Task 29: understand different merge patterns: left, right, inner, outer


# consider the following df "european_champions_title"
european_champions_title = pd.DataFrame({
    "full name": ["Liverpool FC", "FC Barcelona", "Real Madrid CF",
                  "Paris Saint-Germain FC", "SSC Napoli"],
    "UEFA titles": [6, 5, 14, 0, 0],
    "country": ["England", "Spain", "Spain", "France", "Italy"]},
    index=["LIV", "FCB", "RMD", "PSG", "NAP"])

print(european_champions_title)

# Task 30: use pd.merge to left-merge the df "clubs" and "european_champions_title"
# use the index for each df as the join key, name the new df as "new_dataset_left"

print(new_dataset_left)

# Task 31: use pd.merge to right-merge the df "clubs" and "european_champions_title"
# use the index for each df as the join key, name the new df as "new_dataset_right"

print(new_dataset_right)

# Task 32: use pd.merge to inner-merge the df "clubs" and "european_champions_title"
# use the index for each df as the join key, name the new df as "new_dataset_inner"

print(new_dataset_inner)

# Task 33: use pd.merge to outer-merge the df "clubs" and "european_champions_title"
# use the index for each df as the join key, name the new df as "new_dataset_outer"

print(new_dataset_outer)


# consider a new df
more_club_info = pd.DataFrame({
    "points": [1908, 2022, 2001, 2014, 2062],
    "stadium": ["Anfield", "Spotify Camp Nou", "Santiago Bernabeu", "Le Parc des Princes", "Diego Maradona"],
    "country": ["England", "Spain", "Spain", "France", "Italy"]},
    index=["LIV", "FCB", "RMD", "PSG", "NAP"])

print(more_club_info)


# Task 34: use pd.merge to inner-merge the df "clubs" and "european_champions_title"
# use the column "points" in each df as the join key, name the new df as "new_dataset_points"

print(new_dataset_points)

# !Caveat!: non-unique join keys could be problematic

# Task 35: use pd.merge to inner-merge the df "clubs" and "european_champions_title"
# use the column "country" in each df as the join key, name the new df as "new_dataset_points"

print(new_dataset_country)


# Task 36: Export the DataFram new_dataset_outer as a csv file
# name it as "test_dataset.csv"
new_dataset_outer.to_csv()