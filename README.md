# Clothes Similarity Project

## Overview

This project features two Convolutional Neural Network (CNN) models designed to calculate the similarities between two images of clothing.

### Models

| Model Type         | Categories                                                                 |
|--------------------|----------------------------------------------------------------------------|
| **Clothing Type**  | T-shirts, Pants, Sweatshirts, Dresses, Hats, Shoes                         |
| **Shape**          | Circles, Kites, Parallelograms, Rectangles, Rhombuses, Squares, Trapezoids, Triangles |
| **Color**          | Black, Blue, Brown, Cyan, Green, Grey, Orange, Pink, Purple, Red, White, Yellow |

The program averages the cosine similarity scores from each model to compare two images.

### Usage

You can run the program using:
- **Docker**: Simplifies setup and ensures consistency.
- **Training Scripts**: Train the AI with your own data.

Refer to the in-file documentation for detailed parameters and function descriptions.

Models and plots are available [here](https://drive.google.com/drive/folders/16XzrZ49q-yK1KmiGvbj-Nu7ORWCz2PPq?usp=drive_link).

## Installation

Install the required packages:

```bash
pip install --no-cache-dir -r requirements.txt
```

## Running the app

```bash
docker build -t clothes-similarity . 
docker run -it -p 8080:5000 clothes-similarity
```