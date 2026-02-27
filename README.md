## AADHAAR ENROLLMENT ANALYSIS

## DATA REVIEW
* while cleaning the data we found that states_count = 56 which should be 36 so after removing spaces, upper/lower case and handeling the spelling errors by using dictionaries we got the correct states_count = 36.

* while cleaning the data we found that districts_count = 945 which should be around 787 since it is dificult to get the correct district names in the same way as states(using dictionary), so we are thinking of matching it with the 2025 state wise disctrict counts to get the correct districts_count for this dataset or the pincodes.
* No we were wrong about pincodes as there are more pincodes(19509) than the districts(945)  (without cleaning)