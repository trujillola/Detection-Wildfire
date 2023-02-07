# Forest-fire-detection

## Project purpose

format image 224x224x3

## Project Architecture
<ul>
  <li>dataset<ul>
  <ul>
    <li>data
      <ul>
        <li>Fire-Detection-Image-Dataset-master</li>
        <li>Fire-Kaggle</li>
        <li>FIRE-SMOKE-DATASET</li>
        <li>Wildire-forest-fire</li>
      </ul>
    </li>
    <li>data_preprocessed
      <ul>
        <li>annotations
          <ul>
            <li>annotations_dev.csv</li>
            <li>annotations_test.csv </li>
            <li>annotations_train.csv</li>
          </ul>
        </li>
        <li>images<ul><li>dev</li>
            <li>test</li>
            <li>train</li>
          </ul>
        </li>
      </ul>
    </li>
  </ul>
  <li>utils
    <ul>
      <li>Preprocessing_datasets.ipynb</li>
      <li>Dataloader.py</li>
    </ul>
  </li>
  <li>saved_models
    <ul>
      <li>CNN.h5</li>
      <li>FT_ResNet50.h5</li>
    </ul>
  </li>
  <li>results
    <ul>
      <li>Visualisation</li>
      <li>Plots</li>
      <li>predictions.csv<li>
      <li>result_evaluation.csv<li>
    </ul>
  </li>
  <li>models
    <ul>
      <li>Model_CNN.py</li>
      <li>Model_Fine_Tuning.py</li>
    </ul>
  </li>
  <li>Model.py</li>
  <li>_init.py</li>
</ul>


dans annotations_....csv format :  
header : "name","label"
nom_images , label (0 pour pas feu, 1 pour feu)

## Prerequis

pip install codecarbon 
pip install visualkeras

## Dataset 

Sources : 
Images CV : https://images.cv/download/forest_fire/948/CALL_FROM_SEARCH/%22forest_fire%22
Fire kaggle 1 : https://www.kaggle.com/datasets/phylake1337/fire-dataset?resource=download
Fire kaggle 2 : https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data
Fire Kaggle 3 : https://www.kaggle.com/datasets/alik05/forest-fire-dataset

## Modèles

TODO compilation dans build model du model et pas dans le training





