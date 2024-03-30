This is a project on application of finetuning to classify different sizes of classes.
We have developed two separate models to classify weather and bird species based on images.
We have build two apps based on the weights we got from tuning Resnet18 model based on Adam optimizer and CrossEntropyLoss.

In our first model we used 6862 images to predict 11 types of weather. Our model predicts weather truly with 85% chance.

In our second model we used 11788 images to classify birds into 200 classes. To improve our results we finetuned different layers of Resnet model.
Our model detect birds species truly with 90% chances given that birds are from 200 trained birds set.
