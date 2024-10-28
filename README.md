# Clothes Likeliness


## Description
This project contains 2 CNN (Convultional Neural Network) model that allows the program to compute the similarities between 2 images of clothes.

The model are the following:
- Clothes model: it classes the images between 4 type of clothes: t-shirt, pants, sweatshirt and dress
- Shapes model: it classes the images between 8 type of shapes: circle, kite, parallelogram, rectangle, rhombus, square, trapezoid and triangle

The program uses the model one after the other and add the prediction stat to the previous one before applying a cosine similarity to compare 2 images.

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

You can find models and plots of each model [here](https://drive.google.com/drive/folders/1z_0ZIU9b4VxBl2KUg2LhBFKb8ci8yM3u?usp=drive_link)

## Installation

```bash
$  pip install --no-cache-dir -r requirements.txt
```

## Running the app

```bash
$ flask run --host=0.0.0.0 --reload
```


### Developers

| [<img src="https://github.com/Azzzen.png?size=85" width=85><br><sub>Axel Zenine</sub>](https://github.com/Azzzen) | [<img src="https://github.com/ErwanSimonetti.png?size=85" width=85><br><sub>Erwan Simonetti</sub>](https://github.com/ErwanSimonetti) | [<img src="https://github.com/BlanchoMartin.png?size=85" width=85><br><sub>Martin Blancho</sub>](https://github.com/BlanchoMartin) | [<img src="https://github.com/JulietteDestang.png?size=85" width=85><br><sub>Juliette Destang</sub>](https://github.com/JulietteDestang) | [<img src="https://github.com/TdeBoynes.png?size=85" width=85><br><sub>Timothée De Boynes</sub>](https://github.com/TdeBoynes) | [<img src="https://github.com/yanntcheumani.png?size=85" width=85><br><sub>Yann-Arthur Tcheumani</sub>](https://github.com/yanntcheumani) | [<img src="https://github.com/QuentinCn.png?size=85" width=85><br><sub>Quentin Caniou</sub>](https://github.com/QuentinCn)
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |

## License

Nest is [MIT licensed](LICENSE).

