# ArchiTech - International Architectural Style Classification with Grad-CAM
**ArchiTech** is an image classifier of 45 different architectural styles that implements Grad-CAM. 

https://github.com/user-attachments/assets/bc475ed8-51fe-474b-8a74-9840dbe4e908

## Table of content
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Demonstration](#demonstration)
- [Deployment](#deployment)
- [License](#license)
- [Links](#links)
- [Authors](#authors)



## Dataset

The dataset consists of two main components.  

The first is 4720 images across 20 styles sourced entirely from Wikimedia Commons with their API.  

<details><summary><b>Number of images for each style</b></summary>

- Amazigh architecture: 155
- Architecture of Edo Period: 303
- Architecture of the Joseon Dynasty: 255
- Architecture of the Qing Dynasty: 293
- Balinese architecture: 172
- Brutalist architecture: 256
- Buddhist architecture: 220
- Dravidian architecture: 241
- Hausa architecture: 217
- Jain architecture: 107
- Khmer architecture: 257
- Maya architecture: 337
- Minangkabau architecture: 315
- Moorish architecture: 218
- Mughal architecture: 262
- Ottoman architecture: 324
- Pueblo architecture: 68
- Safavid architecture: 344
- Stalinist architecture: 212
- Swahili architecture: 152
</details>

The latter part is 10113 images across 25 styles and authored by:

- Danci, Marian Dumitru (@dumitrux)
- Xu, Zhe & Zhang, Ya & Tao, Dacheng & Wu, Junjie & Tsoi, Ah. (2014). Architectural Style Classification Using Multinomial Latent Logistic Regression. 10.1007/978-3-319-10590-1_39.  

<details><summary><b>Number of images for each style</b></summary>

- Achaemenid architecture: 392
- American craftsman style: 364
- American Foursquare architecture: 362
- Ancient Egyptian architecture: 406
- Art Deco architecture: 566
- Art Nouveau architecture: 615
- Baroque architecture: 456
- Bauhaus architecture: 315
- Beaux-Arts architecture: 424
- Byzantine architecture: 313
- Chicago school architecture: 278
- Colonial architecture: 480
- Deconstructivism: 335
- Edwardian architecture: 280
- Georgian architecture: 381
- Gothic architecture: 331
- Greek Revival architecture: 523
- International style: 417
- Novelty architecture: 382
- Palladian architecture: 343
- Postmodern architecture: 322
- Queen Anne architecture: 720
- Romanesque architecture: 301
- Russian Revival architecture: 352
- Tudor Revival architecture: 455
</details>


The original dataset was made by Zhe Xu.
According to the paper, the best accuracy they could get in 2014 was nearly 70% accuracy.
[Paper "Architectural Style Classification using Multinomial Latent Logistic Regression" (ECCV2014)](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/ECCV_2014/papers/8689/86890600.pdf)

- [Final Extended dataset](https://www.kaggle.com/datasets/jungseolin/international-architectural-styles-combined)
- [Extended dataset](https://www.kaggle.com/dumitrux/architectural-styles-dataset)
- [Original dataset](https://www.kaggle.com/wwymak/architecture-dataset)



#### Data augmentation

The extended dataset is used to test the model with "real world" images, in this case google images, whicha are not taken in perfect conditions.

To make the original dataset bigger and get better results, data augmentation is used.
Which is achived creating new images, by transforming (horizontal flip, zoom, light, rotate, etc.) the images.

In this case we go from 4979 to 9588 images.

![Data augmentation](./images/data-augmentation.jpg)



## Training

The dataset is split in two:
  * Training set (80%)
  * Test set (20%)



## Results
We got a **77%** accuracy on the International Architectural Styles Dataset.

![Results](https://github.com/user-attachments/assets/4e9520a2-8737-4fe4-bd39-ec870c31e549)

#### Grad-CAM Results

![image](https://github.com/user-attachments/assets/841cb9d0-e60e-4660-8406-911242e2474a)

![image](https://github.com/user-attachments/assets/f13834d5-263b-493a-b917-158fd9b4304d)

![image](https://github.com/user-attachments/assets/6e7ba244-5e60-425c-bd00-546eece9e7f0)

## Demonstration

<details><summary><b>More examples</b></summary>

**Gothic:**

https://github.com/user-attachments/assets/35a1a680-2fb1-42cf-b7b0-b2319a0bf3d2

**Edo Period:**

https://github.com/user-attachments/assets/ed496bd1-15e6-4a54-a814-e5eac1bd05ab

**Minangkabau:**

https://github.com/user-attachments/assets/7758aa80-1a87-4ea7-bb04-a3be7521cc15

**Balinese:**

https://github.com/user-attachments/assets/bec850e6-ad05-4622-a8c0-ad798eb2e010

**Khmer:**

https://github.com/user-attachments/assets/078fef4d-6b11-47a9-bb49-6c1249200ee8

**Maya:**

https://github.com/user-attachments/assets/d2079bfc-65da-4e36-9963-018b504e4891

**Swahili:**

https://github.com/user-attachments/assets/fd298a55-e433-453f-971f-8c163cadc4f2


</details>



## Deployment

Made with [fastai](https://www.fast.ai) library.
Source code on [src](./src/architectural-style-recognition.ipynb). It can be run in [Google Colab](colab.research.google.com)

The [app](./app) folder contains the application ready to deploy on [Render](https://render.com).



### Run the app locally

The app can be runned locally.

First of all, you have to check all the [Requirements](./requirements.txt), then run this command in your terminal:

    python app/server.py serve

Or in python3:

    python3 app/server.py serve


## License

MIT License



## Links

* [fastai](https://www.fast.ai)
* [Render](https://render.com)
* [Google Colab](colab.research.google.com)

## Acknowledgements
For this project, we built upon [Marian-Dumitru Danci](https://github.com/dumitrux/architectural-style-recognition)’s model with our implementation of ResNet152. We also built upon [Kazuto Nakashima](https://github.com/kazuto1011/grad-cam-pytorch?tab=readme-ov-file##references)’s Grad-CAM for ResNet152 with our implementation of Grad-CAM. The dataset that we used to train our model is a combination of our original dataset and the works of [Danci](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset) and [Xu et al.](https://doi.org/10.1007/978-3-319-10590-1_39), which altogether contain 45 classes and 14,833 images of buildings. 

## Authors
- Jung, Seolin ([@seolinjung](https://github.com/seolinjung))
- Ference, Jill ([@JillFerence](https://github.com/JillFerence))
- Sengkeo, Puthypor ([@Puthyporsk](https://github.com/Puthyporsk))
