from keras import layers, models

class UNet():
    """
    Architecture modified from the implementations below to include batch normalization, dropout
    and use valid padding (completely avoid border issues):
	https://github.com/zhixuhao/unet/blob/master/unet.py
    https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
    https://github.com/aupilot/kotiki/blob/master/unet2.py
	https://www.kaggle.com/cjansen/u-net-in-keras
    """

    def __init__(self):
        print ('build UNet ...')

    # Compute border sizes to be cropped out before concatenating the feature maps (horizontal connections in UNet)
    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self, img_shape, num_class, dropout):

        concat_axis = 3
        inputs = layers.Input(shape=img_shape)

        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv1_1')(inputs)
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(conv1)
        conv1 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(pool1)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(conv2)
        conv2 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(pool2)
        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(conv3)
        conv3 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        # TODO: Uncomment these lines to add the traditional depth to the UNet
        # conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(pool3)
        # conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(conv4)
        # conv4 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv4)
        # pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        # TODO: Uncomment these lines to add the traditional depth to the UNet
        # conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='valid')(pool4)
        # conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='valid')(conv5)
        # conv5 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv5)
        conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(pool3)
        conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(conv5)
        conv6 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv5)

        # TODO: Uncomment these lines to add the traditional depth to the UNet
        # up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
        # ch, cw = self.get_crop_shape(conv4, up_conv5)
        # crop_conv4 = layers.Cropping2D(cropping=(ch,cw))(conv4)
        # up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
        # conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(up6)
        # conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(conv6)
        # conv6 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv6)

        up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = layers.Cropping2D(cropping=(ch,cw))(conv3)
        up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(up7)
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(conv7)
        conv7 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv7)

        up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = layers.Cropping2D(cropping=(ch,cw))(conv2)
        up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(up8)
        conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(conv8)
        conv8 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv8)

        up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = layers.Cropping2D(cropping=(ch,cw))(conv1)
        up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(up9)
        conv9 = layers.Dropout(dropout)(conv9)
        conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(conv9)
        conv9 = layers.BatchNormalization(axis=-1, center=True, scale=True)(conv9)

        conv10 = layers.Conv2D(num_class, (1, 1), activation='softmax', name='output_layer')(conv9) # 'softmax' for multiclass case, to be combined with 'categorical_crossentropy'

        model = models.Model(inputs=inputs, outputs=conv10)

        return model