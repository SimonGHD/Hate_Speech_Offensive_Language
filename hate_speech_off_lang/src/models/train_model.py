def Network_training(features_lg, labels_hate):
    import os

    cwd_nb = os.getcwd()

    cwd = os.path.dirname(cwd_nb)
    from tensorflow import keras
    print(keras.__version__)
    import tensorflow as tf

    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # let's see what compute devices we have available, hopefully a GPU
    sess = tf.Session()
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    from keras.models import Sequential
    from keras.layers import Dense

    model_hate_speech_MLP_lg = Sequential()
    model_hate_speech_MLP_lg.add(Dense(300, input_dim=300, kernel_initializer='normal', activation='relu'))
    model_hate_speech_MLP_lg.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model_hate_speech_MLP_lg.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model_hate_speech_MLP_lg.add(Dense(1, kernel_initializer='normal'))
    print(model_hate_speech_MLP_lg.summary())

    model_hate_speech_MLP_lg.compile(optimizer='adam', loss='mean_squared_error')
    from time import time

    start = time()
    history = model_hate_speech_MLP_lg.fit(features_lg, labels_hate, epochs=100, batch_size=64, verbose=1)
    end = time()

    model_hate_speech_MLP_lg.save(cwd + '/models/hate_speech_model_lg_word_embed.h5')
    return model_hate_speech_MLP_lg
