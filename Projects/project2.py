# Section 1 - Preparation

import json

data = json.load(open('project2_data-1.json'))

# Section 2 - extract information from the given dataset, create necessary python objects for further use.

guyprefers = data['men_preference']

galprefers = data['women_preference']

free_guy = list(guyprefers.keys())

engage_book = dict()

guypreference = guyprefers.copy()

galpreference = galprefers.copy()

# Section 3 - Implement the Gale-Shapley algorithm to solve the problem

# keep interating while there are still available men
while len(free_guy) != 0:
    
    # take the first available man from the list
	guy = free_guy.pop(0)
 
	# go through each woman in the guy's preference list
	for gal in guypreference[guy]:
		# if the woman he prefers has not already been proposed to, they will become engaged
		if gal not in engage_book:
			engage_book.update({gal: guy})
			break
		# if the woman he prefers has already been proposed to, check to see who the woman prefers
		else:
			fiance = engage_book[gal]
			
			# check to see if the woman prefers the guy she is already engaged to or the one who is proposing to her
			if galpreference[gal].index(fiance) > galpreference[gal].index(guy):
				# if the woman prefers the guy proposing to her, change who she's engaged to and add her ex-fiance back into the free guys list
				engage_book[gal] = guy
				free_guy.append(fiance)
				break

	# if the guy gets rejected by every woman and does not end up getting engaged, add him back into the list
	if guy not in list(engage_book.values()):
		free_guy.append(guy)