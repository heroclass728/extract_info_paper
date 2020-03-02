# DetectPage

## Overview

This project is to extract the information(total marks, name and examine number) from the examine paper and save it into the database. 
To extract the necessary information from the paper, Google Vision API is used for OCR. Also, since there is a table in 
the part which contains the total marks, so the paper image is preprocessed by OpenCV. And every time the paper comes 
into the camera, it needs that the information is extracted, so multi threading technology is necessary for this project.
There are lots of handwritten digits in the paper, but Google Vision API can't extract the exact digit from them. 
Therefore, after training handwritten digits model, the accuracy to detect it can be improved using the trained model. 
At last, this project  supports the smart GUI using Kivy framework.

## Project Structure

- apply_ocr
    
    * The part to get the OCR result from Google Vision API
    * The main part to extract the necessary information from the paper
    * The part of handwritten digits detection
    * The part to crop the necessary part of the paper image

- build_gui

    This project consists of two GUIs, one is to show the paper's coming in and out captured by the camera in the real 
    time, and the result to extract from the paper. And the other is to show the database, where the information is saved.

- manage_database

    This project uses Mysql database to read, insert, update and delete the information on GUI.

- model

    The model to detect the handwritten digits.

- source
    
    * Google Vision key
    * tThe paper image or video stream to detect

- utils

    * The image processing tool
    * The part to train the model with minist dataset
    * several constants

- main
    
    The main execution part

- requirements.txt

    The several libraries for this project

## Project Installation

- Environment

    Ubuntu 18.04, Python 3.6, Mysql newest version
    
- Library Installation

    ```
        pip3 install -r requirements.txt
    ```

## Project Execution

- Please connect web camera to your pc and go ahead to this project directory, run the following command in the terminal.

    ```
        python3 main.py
    ```
