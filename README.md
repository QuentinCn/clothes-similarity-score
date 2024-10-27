# Clothes Likeliness Final Project

This project contains 2 CNN (Convultional Neural Network) model that allows the program to compute the similarities between 2 images of clothes.

The model are the following:
 - Clothes model: it classes the images between 4 type of clothes: t-shirt, pants, sweatshirt and dress
 - Shapes model: it classes the images between 8 type of shapes: circle, kite, parallelogram, rectangle, rhombus, square, trapezoid and triangle 

The program uses the model one after the other and add the prediction stat to the preious one before applying a cosine simimilarity to compare 2 images.

In order to use the program you just have to run the main.py file. When you do, ensure that you completed the parameters in the way you want the program to work.

This is what will be run by default:

````python
main_model(train=False, evaluate=False, predict=False, specific_set='clothes', epoch_try=500, safe_data_load=True, show_activation=True)
main_model(train=False, evaluate=False, predict=False, specific_set='shape', epoch_try=50, safe_data_load=True, show_activation=True)
compute_similarities()
````

- The two first line will act on the models (clothes, shapes) and the third line will compute the similarities between all the images contained in the dataset/tester folder.
- The third line will compute the similarities between all files of the dataset/tester folder.

You can refere to the documentation inside the files to know more about the parameters and goals of the functions and class.

You can find models and plots of each model [here](https://drive.google.com/drive/folders/1TJnZvZXeKKFFedgwNYnbjCFBoJe1VDro?usp=drive_link).