import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Reshape, Bidirectional, LSTM

def create_crnn_model(input_shape=(128, 128, 1), num_classes=8):
    """
    Create a Convolutional Recurrent Neural Network (CRNN) for audio classification
    """
    model = Sequential([
        # CNN part
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2,2)),  # 64x64
        
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),  # 32x32
        
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),  # 16x16
        
        # Reshape for RNN
        Reshape((256, 128)),  
        
        # RNN part
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        
        # Classification head
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model