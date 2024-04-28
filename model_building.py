'''
This script can be used as a blueprint for model creation
on TensorFlow. There are more easier methods to create them
with lesser code

Here we create MobileNetV2 feature extractor
Perform a 2D average pooling
flatten the output
pipe it to 2 dense layers
Use a 4 neuron dense layer as output

The 4 neuron dense layer is in this case the bounding box '''

def feature_extractor(inputs):
    # Create a mobilenet version 2 model object
    mobilenet_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
    alpha=1.0,
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax')


    # pass the inputs into this modle object to get a feature extractor for these inputs
    feature_extractor = mobilenet_model(inputs)

    # return the feature_extractor
    return feature_extractor

def dense_layers(features):

    # global average pooling 2d layer
    x = tf.keras.layers.GlobalAveragePooling2D()(features)


    # flatten layer
    x = tf.keras.layers.Flatten()(x)

    # 1024 Dense layer, with relu
    x = tf.keras.layers.Dense(1024,activation='relu')(x)

    # 512 Dense layer, with relu
    x = tf.keras.layers.Dense(512,activation='relu')(x)

    return x

'''Output Layer'''
def bounding_box_regression(x):
    # Dense layer named `bounding_box`
    bounding_box=tf.keras.layers.Dense(units=4,activation='relu')(x)
    bounding_box_regression_output = bounding_box

    return bounding_box_regression_output

'''Model Building'''
def final_model(inputs):
    # features
    feature_cnn = feature_extractor(inputs)

    # dense layers
    last_dense_layer = dense_layers(feature_cnn)

    # bounding box
    bounding_box_output = bounding_box_regression(last_dense_layer)

    # define the TensorFlow Keras model using the inputs and outputs to your model
    model = tf.keras.Model(inputs=inputs,outputs=bounding_box_output)
	
	return model

'''Compilation of the model'''
def define_and_compile_model():

    # define the input layer
    inputs = tf.keras.Input(shape=(244,244,3))

    # create the model
    model = final_model(inputs) 

    # compile your model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9), loss="mse", metrics=["mae"])

	return model

'''Summary of the model'''
model = define_and_compile_model()
model.summary()

