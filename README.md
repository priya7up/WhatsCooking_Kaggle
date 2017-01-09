This project gives a simple solution to Kaggle's What's Cooking? competition: https://www.kaggle.com/c/whats-cooking
Please read the details of the competition and the description of the data on the Kaggle link. In brief, the training data contains a bunch of recipes with various ingredients. The recipes are labeled with the type of cuisine they belong to. The test data also contains a bunch of recipes with various ingredients and the objective is to determine which cuisine the recipe belongs to.

The data is not clean. Some ingredients begin with a capital letter while others are lowercase. Several ingredients contain brand names such as 'Progresso chicken soup'. There are many different ways in which the same ingredient could be specified in a recipe, for example, garlic, garlic clove, clove garlic etc. Similarly, you will find tomato as well as tomatoes in the recipes so several ingredients are listed as singular while other have plurals. There are many spelling errors in the recipes as well. The ingredients might also have actions included in them, for example, noodles boiled and drained, onions chopped finely etc. As one can imagine, a thorough data clean up could help with several of these issues.

However, I wanted to take a very simple approach to solve this competition at the cost of high accuracy. I opted to use naive bayes algorithm (you can read about it here https://en.wikipedia.org/wiki/Naive_Bayes_classifier):

prob{cuisine|ingredients} = prob{ingredients|cuisine} * prob{cuisine} / prob{ingredients}

prob{cuisine|ingredients} = prob{ingredient_1|cuisine}*prob{ingredient_2|cuisine}....*prob{ingredient_N|cuisine} *                                                 prob(cuisine) / (prob{ingredient_1}*prob{ingredient_2}.....prob{ingredient_N})
                                                              
There are a few data prep tricks I did use:
1. Changed everything to lowercase
2. Changed everything from plural to singular
3. Since salt was such an unusually highly occurring ingredient, it was acting as an outlier so I removed salt from all the recipes
4. Used ngrams (https://en.wikipedia.org/wiki/N-gram) to improve the accuracy of the results. You can play around with various values of the n_grams variable and see how the accuracy changes.

Another extremely important trick that seems to make a huge difference in accuracy is penalizing the rare ingredients. When calculating the probability of an ingredient given cuisine, if the ingredient has never occurred in that given cuisine in the training data, then that ingredient is not included in the calculation of the ingredient probability (denominator of the equation above) either. It is as if that ingredient does not exist when talking about that given cuisine. This improves the results by a factor of two.

The code is in the Whats_Cooking_Naive_bayes file and is fairly straight-forward. There are some plots included in the Data_Visualization file that are helpful in understanding the data better before you start on the competition.

The goal of this project was to solve the What's Cooking challenge with the simplest solution that can be coded up pretty fast and still get a reasonable (~75 % accuracy) result. I purposefully did not use the in-built Naive Bayes classifier in sklearn because the goal was to solve the problem from first principles. Some ideas to improve the accuracy of the results are: we can try to clean up the data better to take care of some of the issues described above, we can use a different model such as a neural network or we can stack various models for better classification.
