import tensorflow as tf
from tensorflow.keras import layers, models

def build_vibe_model(input_shape=(128, 128, 3), stats_dim=5, num_classes=3):
    spectrogram_input = layers.Input(shape=input_shape, name="spectrogram")

    x = layers.Conv2D(32, (3, 3), padding='same')(spectrogram_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(192, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    stats_input = layers.Input(shape=(stats_dim,), name="audio_stats")
    s = layers.Dense(32, activation='relu')(stats_input)
    s = layers.Dropout(0.2)(s)

    combined = layers.Concatenate()([x, s])
    combined = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(combined)
    combined = layers.Dropout(0.4)(combined)
    output = layers.Dense(num_classes, activation='softmax')(combined)

    model = models.Model(inputs=[spectrogram_input, stats_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# To meet Syllabus Item #3: Plot the model
# tf.keras.utils.plot_model(model, to_file='model_arch.png', show_shapes=True)

if __name__ == "__main__":
    model = build_vibe_model()
    model.save("advanced_vibe_meter.keras")
    print("Model saved to advanced_vibe_meter.keras")
