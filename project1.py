# Problem 0
import time

# start time for the program for problem 10
start_time = time.time()

# Problem 1 - Generate a list with numbers from 1 to 20, name the list as list1.
list1 = list(range(1, 21)) # create a list from 1 to 20

# Problem 2 - Print the first 5 elements of this list.
print('Problem 2 Answer:',list1[:4]) 

# Problem 3 - Replace the last entry of the list with 100, and print the list again.
list1[-1] = 100
print('Problem 3 Answer:',list1)

# Problem 4 - Sort the list from largest to smallest element, and print the list again.
list1.sort(reverse=True)
print('Problem 4 Answer:',list1)

# Problem 5 - Generate a new list with entries from 14 to 40 with step size 2, name it as list2.
list2 = list(range(14, 41, 2))
print('Problem 5 Answer:',list2)

# Problem 6 - Write a loop, dividing the first 10 entries of list2 by 5, keep the rest of the 
# 			  list unchanged, and store the result into list3.
list3 = list()
for idx in range(len(list2)):
	if idx < 10:
		list3.append(list2[idx]/5)
	else:
		list3.append(list2[idx])
		
# Problem 7 - Given the dictionary hrbook, print the value associate with the key "emp2".
hrbook= { 
	'emp1': {'name': 'John', 'salary': 7500}, 
	'emp2': {'name': 'Emma', 'salary': 8100}, 
	'emp3': {'name': 'Brad', 'salary': 6500} 
	}
print('Problem 7 Answer:',hrbook['emp2'])

# Problem 8 - Add a new record to the hrbook, the key is emp4, value is {'name': 'Misty', 'salary': 7700}
hrbook.update({'emp4': {'name': 'Misty', 'salary': 7700}})

# Problem 9 - Use loop and conditional branching to do the following:
#					for those whose salary is lower than 7000, replace the salary with 7000, 
# 					for those whose salary is between 7000 and 8000, replace the salary with 8000,
# 					for those whose salary is higher than 8000, replace the salary with 8200. 
for key, value in hrbook.items():
	if value['salary'] < 7000:
		value['salary'] = 7000
	elif 7000 <= value['salary'] <= 8000:
		value['salary'] = 8000
	else:
		value['salary'] = 8200
  
# Problem 10 - use the time module to "time" your work. print out how long it takes to run the whole script.
end_time = time.time()
print('Problem 10 Answer:',f"Time taken - {end_time - start_time} seconds")