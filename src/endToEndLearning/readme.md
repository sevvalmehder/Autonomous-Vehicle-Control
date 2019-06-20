### End to End Learning  

The model in that study based on Nvidia [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). The given method in the paper is a regression solution. But classification is used in this study. Steering angles are divided to te 19 class. When predicted class is 1 then applied 18 degree to the steer. 36 degree for class 2, -18 degree for the class 10 and so on.  

driving_dataset has two folder as test and train. These two file has to include steering angles in data.txt file.  