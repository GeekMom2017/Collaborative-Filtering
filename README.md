This code is a Python implementation of a user-based collaborative filtering model where the features of each item and users' preferences for those features are learned implicitly through matrix factorization and gradient descent.

Currently, the code does not handle predictions for new users. To add this functionality, we can start with an average rating for each item by all other users as a first prediction for new users.

Data from https://grouplens.org/datasets/movielens/100k/

Main script: codes/Collaborative Filerting.ipynb
Support functions: codes/helper.py

Data is stored in ml-100k/
Checkpoints: checkpoints/


