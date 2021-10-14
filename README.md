# INSTAGRAM DEEP TARGETING


---

How to navigate this repo:
- proposal_ig.pdf: my initial proposal for an image classification project 
- mvp_ig.pdf: mvp presented two days before the final project submission
- final folder: all files used for the final submission
  1. 01_ig_eda.ipynb : loaded data and conducted EDA on the 101,000 images
  2. 02_ig_image_classification_baseline.ipynb : built a random forest baseline after resizing, grayscaling and flattening the 101,000 images 
  3. 03_ig_image_classification_mini.ipynb : applied transfer learning with ResNet50 to classify 3,000 images across 3 separable food classes
  4. 04_ig_image_classification_full_dataset.ipynb : applied transfer learning with ResNet50 to classify 101,000 images across 101 food classes on 256 x 256 resolution images
  5. 05_ig_image_classification_full_dataset_high_res.ipynb : applied transfer learning with ResNet50 to classify 101,000 images across 101 food classes on 512 x 512 resolution images. This is currently the final version of my model.
  6. instagram_deep_targeting.pdf : final presentation

---

## ABSTRACT
As a Data Scientist at Instagram, I have been tasked with finding a way to increase Instagram Ad Revenues. To achieve this goal, I have decided to build a targeting engine, called *deep targeting*, based on user-generated content. 

In particular, I have created my prototype on food-related businesses (e.g. restaurants, subscription boxes, food producers). The impact hypothesis is that by classifying food images shared by users on Instagram and understanding user preferences, Instagram will provide advertisers will a simultaneously data-driven and user-driven tool to target prospects with ads in a more precise and relevant way. Equipped with a more accurate targeting algorithm, advertisers will be able to boost their revenues and, in turn, will likely increase their marketing spend and lifetime on Instagram.

After loading and organizing the [Food-101](https://www.tensorflow.org/datasets/catalog/food101) image data, I have run a random forest classifier as a baseline model. Then, I have applied transfer learning and built upon a ResNet50 CNN first on 3 classes only, then on the full 101 food classes and, finally, on the 101 food classes with high-resolution images.

## DATA
I have collected food image data from the [Food-101](https://www.tensorflow.org/datasets/catalog/food101) dataset, containing a total of 101,000 images labeled according to 101 food categories. 

For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise - which comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.

The image dataset, in its original form, is structured by food class - hence, all 1,000 images (250 test + 750 train) corresponding to the same food class are included in the same folder named as the food class they represent. In order to make this dataset easily processable by my CNN, I had to read in the labels included in the *test.json* and *train.json* in the metadata folder and create - through the os library in jupyter notebook - two additional directories, *test* and *train*, that themselves contained the 101 folders corresponding to the food classes labels.

Finally, the last challenge from a data handling standpoint was to transfer the image dataset to Google Drive in order to be able to easily access it via Colab. In this step, I found out that the fastest way to execute the data transfer was to zip the *test* and *train* directories locally, upload them onto Google Drive, and unzip them in the cloud.

## DESIGN
As reflected in the notebooks included in the final folder of this repo, this project is designed around five main components:
1. **Data Loading & EDA** : prepare and understand the image data its specificities 
2. **Model Baseline** : after preprocessing and flattening the images, run a random forest classifier as a simple baseline
3. **CNN on 3 food classes** : build a first CNN prototype on only 3,000 images and 3 food classes
4. **CNN on 101 food classes** : apply the CNN tested in phase 3 to the 101,000 256 x 256 resolution images and 101 classes
5. **CNN on 101 food classes and hi-res images** : apply the CNN tested in phase 4 to the 101,000 512 x 512 resolution images and 101 classes

As a last step, in light of the results from my analysis, I have generated a few business insights and model development ideas that I have summarized in the final presentation. 

## ALGORITHM
As a first step, I loaded and structured the image data as needed to run a CNN. Then, I investigated and visualized the images to get a better understanding of the type and quality of images I was working with. Specifically, I noticed that the images presented some challenges: poor and inconsistent quality, multi-class images (i.e. images that contained food items from multiple classes) and intra-class similarity (e.g. *steaks* photos were very similar to *filet mignon* photos).

Then, I tackled the **Model Baseline** phase by first resizing, grayscaling and flattening my images into multiple vectors that I then concatenated into a pandas dataframe. Then, I used sklearn to run a random forest classifier that, unsuprisingly, performed pretty poorly in terms of accuracy and F1 score.

After these first two steps I started working with fast.ai and pytorch to build a CNN and classify the images as accurately as possible. This phase, uniquely focused on building a neural network, is split in three sub-phases.
First, I decided to test the CNN on three separable food classes: doughnuts, pizza, and hamburger. I applied transfer learning with ResNet50, a CNN that is known to be particularly effective at classifying images. In addition, in order to boost generalizability and regularization of my dataset I applied both composite image augmentation and mixup image augmentation (see more details in notebook 03). To fine tune the model and enhance accuracy, I adopted Leslie Smith's method of [one cycle policy](https://arxiv.org/abs/1803.09820), as well as discriminative learning rates and test time augmentation.
Once the model was fine-tuned and tested, I scaled the model to the 101,000 images on Google Colab (using both GPU and high-RAM) and applied it both to 256 x 256 resolution images (notebook 04) and 512 x 512 resolution images (notebook 05). This last model is my final one and presents a top-1 accuracy of 0.901 and a top-5 accuracy of 0.984. 

## TOOLS
- Os to interact with the operating system from jupyter lab and restructure my dataset into multiple directories
- Pandas and numpy for data manipulation
- Fastai, keras and pytorch for CNN 
- Fastai, seaborn and sklearn for data visualization
- Google Colab and related GPU to scale the model to 101 food classes and 101,000 images 

## COMMUNICATION
A slide deck is included in this repo. 

In addition, see below a few t-SNE plots related to all 101 classes, a few similar classes and a few dissimilar classes.<br/>

**1. t-SNE on all food classes**

<img width="723" alt="tsne_all" src="https://user-images.githubusercontent.com/68084582/125006524-5ab61f80-e02c-11eb-8fea-1ef2667da7ac.png">

**2. t-SNE on most confused food classes** 

<img width="782" alt="tsne_similar" src="https://user-images.githubusercontent.com/68084582/125006375-fd21d300-e02b-11eb-8d19-228a120cb24e.png">

**3. t-SNE on dissimilar food classes** 

<img width="783" alt="tsne_dissimilar" src="https://user-images.githubusercontent.com/68084582/125006400-0b6fef00-e02c-11eb-87b2-d82891a52a62.png">



