# ResNet-Pytorch-Face-Recognition
Using Pytorch to implement a ResNet50 for Cross-Age Face Recognition<br>
Generally speaking, Pytorch is much more user-friendly than Tensorflow for academic purpose.

Prepare Dataset and Environment
====
Trainable ResNet50 using Python3.5 + Pytorch <br>
DataSet: Cross-Age Celebrity Dataset[(CACD)](http://bcsiriuschen.github.io/CARC/) <br>
By default, you should put all the CACD images under "./CACD2000/". If your dataset is in another place, add "--root-path your_path" when you run main.py. <br>
You don't need to crop image into 224*224*3 size anymore, pytorchvision provide convenient "transforms" to do so.

Explanation of Each File
====
1. main.py is just used to control parameters, it doesn't contains any useful details.
2. ResNet.py & VGG.py are the cores of the project, they have the implementation of Networks.
3. train.py contains the details of training process.
4. data.py maintains a Class to generate CACD data class, which is very different with Tensorflow and quite useful.
5. In /model/params.pkl, we give a pretrained model learnt through default setting (change number of epoch to 30)

Training Part
====
1. Run main.py directly, there are some options and parameters you can modify, pleace check the details of main.py.
2. About "--model", there are 3 options: resnet50, resnet101, vgg16. Although I also implemented VGG class, but I didn't check whether it's working or not, so the default option is resnet50.
3. Labels and Image names are loaded from "./data/label.npy" and "./data/name.npy".<br>
   Pytorch provide a very useful approach to create datasets. Please check data.py
4. Label is range from [1, LABELSNUM], to make correct classification, we should change it to [0, LABELNUM-1]
5. If you want to load a existing model, you should run the following "python main.py --model-path your_path --pretrained 1".

Evaluation 
====
Since I just want to get used to Pytorch. I didn't prepare a evaluation method, you can make your own if you like.
