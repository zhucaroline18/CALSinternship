# CALSinternship
An exploration of the value of machine learning within nutrition

## Project Objectives
Lots of data in the nutrition labs have gone unused- deemed insignificant by statistical analysis. However, perhaps there is a missing correlation unseen because of it's relations to other unseen factors that go beyond just the addition of some ingredients. Every diet comes out with some output and maybve the nutritional profile rather than the ingredient profile of a diet would provide a better pathway to finding connections. This is also powerful because every experiment can be a part of this because every diet has a nutrient profile, rather than just the few experiments that use x ingredients. 

We will aim to create a machine learning model that can predict health results based on the nutritional profile of a diet (such as the various carbohydrates, proteins and amino acids, lipids and fatty acids, vitamins, and minerals).

## Project Methods
This project can be taken in 3 main stages 
- building up the database
- building up the neural network
- incorporating the neural network into a Large Language Model

# Project Challenges and Considerations 
A lot of the challenges come in the actual data collection- everyone in the lab organizes their data in a different way making it difficult and time consuming to add the data into the database. A lot of calculation is necessary to calculate the nutrient composition of each diet. In addition, we are currently finding the total nutrient consumed in a lifetime since birth of a chicken (this is find for chickens as we can keep track of the nutrients from birth to 6 weeks, however is probably not viable for many other animals including humans. We may want to consider something like total nutrients consumed in a day) which requires finding average diet intake, calculating the nutrient composition of that diet, and then multiplying by the total intake across 1-3 weeks or 1-6 weeks based on what result we are looking at. Additionally, because broiler chickens are fed different diets from 1-3 weeks compared to 3-6 weeks, it makes it a hassle to calculate the diets from 1-6 weeks. For example, if we are looking at a result measured at week 3, then we would look at the 1-3 week total intake. However, if we are measuing a result at week 6, we consider that an ouput from 1-6 weeks of nutrient intake. So, as you can see, inputting data into the database is an extremely time consumming endeavor. This can be made easier in the future by implementing a standardized system for data collection across the lab so that the work can be distributed across the grad students.

Additionally, another challenge is that a lot of the data has missing datapoints. While most studies will measure growth, that is where the similarities end. Some measure blood, or liver gene data, or breast gene data, or meat quality, etc, or any combination of those. This means there will be loads of missing data and it is difficult to know what to do with the missing data. Most likely what will have to happen is to have a separate model on meat quality, a separate one on blood, etc but there have been other suggestions like inputting an average.

Until the final archetecture can be determined, it is probably best to hold off on creating an official sql database or it will be a hassle to have to change the structure if we deem a change needs to be made. For now, we are working within an excel file that will be converted to CSV however that presents its own challenges as it can be quite laborious for excel to deal with so much data depending on the power of your computer. 

Another challenge that presents itself later in the project has to deal with the LLM part of the project. Currently, we are experimenting with using LLM Agents with tool use, providing the agent with statistical tools to see if it can find relationships between variables in the data. However, we run into a few problems with this. First, the model is llama 3 which is not the most well trained model but it is free to use. Then, we are running into the issue of the model making assumptions. When we ask the model not to make assumptions, it makes too many calls to the API and causes a rate limit exceeded error as well as a content length exceeded error. For larger scales of the project, if we decide to incorporate with LLMs, we may have to use paid version that allow for more calls as well as more accuracy. 

Additionally, another challenge that presents itself is that Dr. Lei wants to stay away from traditional statistical methods. While we can try the tool use with the statistical methods, he ultimately wants it to be more focused on AI concepts. We need to think of more tools that the LLM can use for evaluation of the data.

## Guide to Folders
### [DatabaseBuilding](./DatabaseBuilding)
Relevant Files 
- [access.py](./DatabaseBuilding/access.py): an initial attempt to create a sql access layer
- nutrition.db: the sql database connected to access.py

Note about Progress
- This was an initial attempt to create a database access layer- it proved difficult as there were so many names and also there was too many changes to the architecture to be figured out so I think it's definitely better to be sure about the structure of the input and output files before attempting to create a standardized database access layer
- You will notice in access.py that data is being inserted into the database in 3 steps: First, we insert the diets and their nutrient composition into the diets table. Then, we insert the followers of the diets and the health results of that individual, connecting each follower to the diet that it is eating and noting how much of that diet it eats and inserting that into the DietFollowes table. When extracting information about the diet follower, it will calculate the total nutrients consumed within the time period. It inserts the results associated with that DietFollower into a separate Results table that is linked to the diet follower or individual. Then, there are the individuals which don't follow a diet in the table and will list out all the nutrients consumed and the results. The nutrients consumed will be added to the individuals table while the results will be added to the results table with a foreign key connecting them.
- Again, the architecture is constantly changing so this is not the way I have structured the data anymore. Now, I organize the data like so: [here](https://1drv.ms/x/s!AhTsi-CmQfC04B1ZdRpoXWDRYoeZ?e=6JYybi) where you just have the nutrients consumed on one side and the health results on the other


## link to other resources
[Nutrient Profile Calculator](https://1drv.ms/x/s!AhTsi-CmQfC04U-HmMr5EqZ26ZC4?e=bDB5xT&nav=MTVfe0IyNUNBOUQ1LTM5MkUtNEE0Ny04NjFGLTE5NjA1QjhDMkEwQX0)
