# TensorflowBaseDemo
该Demo演示了，使用Tensorflow训练模型的基本流程

请将以下文件解压到当前目录
- Fnt.tar.gz
- modelAndTFRecord.zip
- 文件目录结构：<br>
![ls](/ls.jpg)


------
> 本文已在公众号`机器视觉与算法建模`发布，转载请联系我。<br>
# 使用TensorFlow的基本流程
本篇文章将介绍使用tensorflow的训练模型的基本流程，包括制作读取TFRecord，训练和保存模型，读取模型。

## 准备
- 语言:Python3
- 库:tensorflow、cv2、numpy、matplotlib
- 数据集:Chars74K dataset 的数字部分
- 网络:CNN
所有代码已经上传至github：https://github.com/wmpscc/TensorflowBaseDemo

## TFRecord
TensorFlow提供了一种统一的格式来存储数据，这个格式就是TFRecord.
``` Python
message Example {  
 Features features = 1;  
};  
  
message Features{  
 map<string,Feature> featrue = 1;  
};  
  
message Feature{  
    oneof kind{  
        BytesList bytes_list = 1;  
        FloatList float_list = 2;  
        Int64List int64_list = 3;  
    }  
};  
```
从代码中我们可以看到， tf.train.Example 包含了一个字典，它的键是一个字符串，值为Feature，Feature可以取值为字符串（BytesList）、浮点数列表（FloatList）、整型数列表（Int64List）。

### 写入一个TFRecord一般分为三步：
- 读取需要转化的数据
- 将数据转化为Example Protocol Buffer，并写入这个数据结构
- 通过将数据转化为字符串后，通过TFRecordWriter写出

#### 方法一
这次我们的数据是分别保存在多个文件夹下的，因此读取数据最直接的方法是遍历目录下所有文件，然后读入写出TFRecord文件。该方法对应文件`MakeTFRecord.py`，我们来看关键代码
``` Python
    filenameTrain = 'TFRecord/train.tfrecords'
    filenameTest = 'TFRecord/test.tfrecords'
    writerTrain = tf.python_io.TFRecordWriter(filenameTrain)
    writerTest = tf.python_io.TFRecordWriter(filenameTest)
    folders = os.listdir(HOME_PATH)
    for subFoldersName in folders:
        label = transform_label(subFoldersName)
        path = os.path.join(HOME_PATH, subFoldersName)  # 文件夹路径
        subFoldersNameList = os.listdir(path)
        i = 0
        for imageName in subFoldersNameList:
            imagePath = os.path.join(path, imageName)
            images = cv2.imread(imagePath)
            res = cv2.resize(images, (128, 128), interpolation=cv2.INTER_CUBIC)
            image_raw_data = res.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_raw_data)
            }))
            if i <= len(subFoldersNameList) * 3 / 4:
                writerTrain.write(example.SerializeToString())
            else:
                writerTest.write(example.SerializeToString())
            i += 1
```
在做数据的时候，我打算将3/4的数据用做训练集，剩下的1/4数据作为测试集，方便起见，将其保存为两个文件。<br>
基本流程就是遍历Fnt目录下的所有文件夹，再进入子文件夹遍历其目录下的图片文件，然后用OpenCV的`imread`方法将其读入，再将图片数据转化为字符串。在TFRecord提供的数据结构中`_bytes_feature'是存储字符串的。<br>
以上将图片成功读入并写入了TFRecord的数据结构中，那图片对应的标签怎么办呢？
``` Python
def transform_label(folderName):
    label_dict = {
        'Sample001': 0,
        'Sample002': 1,
        'Sample003': 2,
        'Sample004': 3,
        'Sample005': 4,
        'Sample006': 5,
        'Sample007': 6,
        'Sample008': 7,
        'Sample009': 8,
        'Sample010': 9,
        'Sample011': 10,
    }
    return label_dict[folderName]
```
我建立了一个字典，由于一个文件下的图片都是同一类的，所以将图片对应的文件夹名字与它所对应的标签，产生映射关系。代码中`label = transform_label(subFoldersName)`通过该方法获得，图片的标签。

#### 方法二
在使用方法一产生的数据训练模型，会发现非常容易产生过拟合。因为我们在读数据的时候是将它打包成batch读入的，虽然可以使用`tf.train.shuffle_batch`方法将队列中的数据打乱再读入，但是由于一个类中的数据过多，会导致即便打乱后也是同一个类中的数据。例如：数字0有1000个样本，假设你读取的队列长达1000个，这样即便打乱队列后读取的图片任然是0。这在训练时容易过拟合。为了避免这种情况发生，我的想法是在做数据时将图片打乱后写入。对应文件`MakeTFRecord2.py`，关键代码如下
``` Python
    folders = os.listdir(HOME_PATH)
    for subFoldersName in folders:
        path = os.path.join(HOME_PATH, subFoldersName)  # 文件夹路径
        subFoldersNameList = os.listdir(path)
        for imageName in subFoldersNameList:
            imagePath = os.path.join(path, imageName)
            totalList.append(imagePath)

    # 产生一个长度为图片总数的不重复随机数序列
    dictlist = random.sample(range(0, len(totalList)), len(totalList))  
    print(totalList[0].split('\\')[1].split('-')[0])    # 这是图片对应的类别

    i = 0
    for path in totalList:
        images = cv2.imread(totalList[dictlist[i]])
        res = cv2.resize(images, (128, 128), interpolation=cv2.INTER_CUBIC)
        image_raw_data = res.tostring()
        label = transform_label(totalList[dictlist[i]].split('\\')[1].split('-')[0])
        print(label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw_data)
        }))
        if i <= len(totalList) * 3 / 4:
            writerTrain.write(example.SerializeToString())
        else:
            writerTest.write(example.SerializeToString())
        i += 1
```
基本过程：遍历目录下所有的图片，将它的路径加入一个大的列表。通过一个不重复的随机数序列，来控制使用哪张图片。这就达到随机的目的。<br>
怎么获取标签呢？图片文件都是`类型-序号`这个形式命名的，这里通过获取它的`类型`名，建立字典产生映射关系。
``` Python
def transform_label(imgType):
    label_dict = {
        'img001': 0,
        'img002': 1,
        'img003': 2,
        'img004': 3,
        'img005': 4,
        'img006': 5,
        'img007': 6,
        'img008': 7,
        'img009': 8,
        'img010': 9,
        'img011': 10,
    }
    return label_dict[imgType]
```

### 原尺寸图片CNN
对应`CNN_train.py`文件
训练的时候怎么读取TFRecord数据呢，参考以下代码
``` Python
# 读训练集数据
def read_train_data():
    reader = tf.TFRecordReader()
    filename_train = tf.train.string_input_producer(["TFRecord128/train.tfrecords"])
    _, serialized_example_test = reader.read(filename_train)
    features = tf.parse_single_example(
        serialized_example_test,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    img_train = features['image_raw']
    images_train = tf.decode_raw(img_train, tf.uint8)
    images_train = tf.reshape(images_train, [128, 128, 3])
    labels_train = tf.cast(features['label'], tf.int64)
    labels_train = tf.cast(labels_train, tf.int64)
    labels_train = tf.one_hot(labels_train, 10)
    return images_train, labels_train
```
通过`features[键名]`的方式将存入的数据读取出来，键名和数据类型要与写入的保持一致。<br>
关于这里的卷积神经网络，我是参考王学长培训时的代码写的。当然照搬肯定不行，会遇到loss NaN的情况，我解决的方法是仿照`AlexNet`中，在卷积后加入LRN层，进行局部响应归一化。在设置参数时，加入l2正则项。关键代码如下
``` Python
def weights_with_loss(shape, stddev, wl):
    var = tf.truncated_normal(stddev=stddev, shape=shape)
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return tf.Variable(var)

def net(image, drop_pro):
    W_conv1 = weights_with_loss([5, 5, 3, 32], 5e-2, wl=0.0)
    b_conv1 = biasses([32])
    conv1 = tf.nn.relu(conv(image, W_conv1) + b_conv1)
    pool1 = max_pool_2x2(conv1)
    norm1 = tf.nn.lrn(pool1, 4, bias=1, alpha=0.001 / 9.0, beta=0.75)

    W_conv2 = weights_with_loss([5, 5, 32, 64], stddev=5e-2, wl=0.0)
    b_conv2 = biasses([64])
    conv2 = tf.nn.relu(conv(norm1, W_conv2) + b_conv2)
    norm2 = tf.nn.lrn(conv2, 4, bias=1, alpha=0.001 / 9.0, beta=0.75)
    pool2 = max_pool_2x2(norm2)

    W_conv3 = weights_with_loss([5, 5, 64, 128], stddev=0.04, wl=0.004)
    b_conv3 = biasses([128])
    conv3 = tf.nn.relu(conv(pool2, W_conv3) + b_conv3)
    pool3 = max_pool_2x2(conv3)

    W_conv4 = weights_with_loss([5, 5, 128, 256], stddev=1 / 128, wl=0.004)
    b_conv4 = biasses([256])
    conv4 = tf.nn.relu(conv(pool3, W_conv4) + b_conv4)
    pool4 = max_pool_2x2(conv4)

    image_raw = tf.reshape(pool4, shape=[-1, 8 * 8 * 256])

    # 全连接层
    fc_w1 = weights_with_loss(shape=[8 * 8 * 256, 1024], stddev=1 / 256, wl=0.0)
    fc_b1 = biasses(shape=[1024])
    fc_1 = tf.nn.relu(tf.matmul(image_raw, fc_w1) + fc_b1)

    # drop-out层
    drop_out = tf.nn.dropout(fc_1, drop_pro)

    fc_2 = weights_with_loss([1024, 10], stddev=0.01, wl=0.0)
    fc_b2 = biasses([10])

    return tf.matmul(drop_out, fc_2) + fc_b2
```
`128x128x3`原图训练过程
![128*128](http://upload-images.jianshu.io/upload_images/7007489-74f9fc7cbc1e1574..jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在验证集上的正确率
![128v](http://upload-images.jianshu.io/upload_images/7007489-99409b0fc6d372f5..jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这里使用的是128*128*3的图片，图片比较大，所以我产生了一个想法。在做TFRecord数据的时候，将图片尺寸减半。所以就有了第二种方法。
### 图片尺寸减半CNN
对应文件`CNN_train2.py`
与上面那种方法唯一的区别是将图片尺寸`128*128*3`改成了`64*64*3`所以我这里就不重复说明了。
`64x64x3`图片训过程
![64*64](http://upload-images.jianshu.io/upload_images/7007489-95fe1cb301dee82c..jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在验证集上的正确率
![64v](http://upload-images.jianshu.io/upload_images/7007489-913bf73a2cdca0d0..jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 保存模型
在`CNN_train.py`中，对应保存模型的代码是
``` Python
def save_model(sess, step):
    MODEL_SAVE_PATH = "./model128/"
    MODEL_NAME = "model.ckpt"
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)

save_model(sess, i)
```
`i`是迭代的次数，可以不填其对应的参数`global_step`

### 在测试集上检验准确率
对应文件`AccuracyTest.py`
代码基本与训练的代码相同，这里直接讲怎么恢复模型。关键代码
``` Python
    ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        #加载模型
        saver.restore(sess, ckpt.model_checkpoint_path)
```
值得一提的是`tf.train.get_checkpoint_state`该方法会自动找到文件夹下迭代次数最多的模型，然后读入。而`saver.restore(sess, ckpt.model_checkpoint_path)`方法将恢复，模型在训练时最后一次迭代的变量参数。

### 查看读入的TFRecord图片
对应文件`ReadTest.py`
如果你想检查下在制作TFRecord时，图片是否处理的正确，最简单的方法就是将图片显示出来。关键代码如下
``` Python
def plot_images(images, labels):
    for i in np.arange(0, 20):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(labels[i], fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()

plot_images(image, label
```
![示例](http://upload-images.jianshu.io/upload_images/7007489-2d8daae279d6549a..jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## 总结
在摸索过程中遇到很多问题，也希望这篇文章能帮助更多人吧。
新手上路，如果有错，欢迎指正，谢谢。


