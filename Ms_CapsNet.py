from keras import layers, models, optimizers, regularizers, constraints
from keras import backend as K
from capsulelayer_keras import Class_Capsule, Conv_Capsule, PrimaryCap1, PrimaryCap2, Length, ecanet_layer, AFC_layer
from data_prepare import readdata
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.layers.merge import add
import scipy.io as scio


def Ms_CapsNet(input_shape, n_class, num_routing):

    x = layers.Input(shape=input_shape)
	#  feature extraction by AFC
    out_afc = AFC_layer(x)
    #  dim_vector is the dimensions of capsules, n_channels is number of feature maps
    Primary_caps1 = PrimaryCap1(out_afc, dim_vector=8, n_channels=4, kernel_size=3, strides=2, padding='VALID')
    Primary_caps2 = PrimaryCap2(out_afc, dim_vector=8, n_channels=4, kernel_size=5, strides=2, padding='VALID')

    Conv_caps1 = Conv_Capsule(kernel_shape=[3, 3, 4, 8], dim_vector=8, strides=[1, 2, 2, 1],
                              num_routing=num_routing, batchsize=args.batch_size, name='Conv_caps1')(Primary_caps1)
    Conv_caps2 = Conv_Capsule(kernel_shape=[3, 3, 4, 8], dim_vector=8, strides=[1, 2, 2, 1],
                              num_routing=num_routing, batchsize=args.batch_size, name='Conv_caps2')(Primary_caps2)
							  						
    Class_caps1 = Class_Capsule(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='class_caps1')(Conv_caps1)
    Class_caps2 = Class_Capsule(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='class_caps2')(Conv_caps2)
	#  fuse the output of class capsule  
    Class_caps_add=add([Class_caps1, Class_caps2]);
	
    out_caps = Length(name='out_caps')(Class_caps_add)

    return models.Model(x, out_caps)


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):

    (x_train, y_train), (x_valid, y_valid) = data

    # callbacks and save the training model
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-test.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss],
                  metrics={' ': 'accuracy'})

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[x_valid, y_valid], callbacks=[tb, checkpoint], verbose=2)

    return model


def test(model, data):
    from sklearn.metrics import confusion_matrix
    x_test, y_test = data[0], data[1]
    n_samples = y_test.shape[0]
    add_samples = args.batch_size - n_samples % args.batch_size
    x_test = np.concatenate((x_test[0:add_samples, :, :, :], x_test), axis=0)
    y_test = np.concatenate((y_test[0:add_samples, :], y_test), axis=0)
    y_pred = model.predict(x_test, batch_size=args.batch_size)
    ypred = np.argmax(y_pred, 1)
    y = np.argmax(y_test, 1)
    matrix = confusion_matrix(y[add_samples:], ypred[add_samples:])
    return matrix, ypred, add_samples


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


if __name__ == "__main__":
    import numpy as np
    import os
    from keras import callbacks

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_class', default=2, type=int)  # number of classes
    parser.add_argument('--epochs', default=50, type=int) 
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)  # learning rate
    parser.add_argument('--windowsize', default=9, type=int) # patch size
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # file path of SAR dataset
    image_file = r'./data/YellowRiverI.mat'
    label_file = r'./data/YellowRiverI_gt.mat'

    data, test_shuffle_number = readdata(image_file, label_file, train_nsamples=1000, validation_nsamples=1000,
                                         windowsize=args.windowsize, istraining=True)
    #  save the index of training samples
    scio.savemat('training_index.mat', {"index":test_shuffle_number})
    (x_train, y_train), (x_valid, y_valid) = (data[0], data[1]), (data[2], data[3])

    # define model
    model = Ms_CapsNet(input_shape=[args.windowsize, args.windowsize, 3],
                    n_class=args.n_class,
                    num_routing=args.num_routing)			
    model.summary()
    # plot_model(model, to_file='model.png')
	
    # model training
	
    train(model=model, data=((x_train, y_train), (x_valid, y_valid)), args=args)


    # model testing 
	
    model.load_weights('./result/weights-test.h5')
    i = 0
    test_nsamples = 0
    RESULT = []
    matrix = np.zeros([args.n_class, args.n_class], dtype=np.float32)
    while 1:
        data = readdata(image_file, label_file, train_nsamples=1000, validation_nsamples=1000,
                        windowsize=args.windowsize, istraining=False, shuffle_number=test_shuffle_number, times=i)
        if data == None:
            OA, AA_mean, Kappa, AA = cal_results(matrix)
            print('-' * 50)
            print('OA:', OA)
            print('AA:', AA_mean)
            print('Kappa:', Kappa)
            print('Classwise_acc:', AA)
            break
        test_nsamples += data[0].shape[0]
        matrix1, ypred, add_samples = test(model=model, data=(data[0], data[1]))
        matrix = matrix1 + matrix
        #matrix = matrix + test(model=model, data=(data[0], data[1]))
        RESULT = np.concatenate((RESULT, ypred[add_samples:]),axis = 0)
        i = i + 1
	# save the final result	
    scio.savemat('final_result.mat', {"final_resule":RESULT})
