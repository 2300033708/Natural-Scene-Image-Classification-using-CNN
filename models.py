from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_intel_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # ✅ Correct paths according to your folder structure
    base_dir = os.path.join('intel_cnn', 'dataset')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    datagen = ImageDataGenerator(rescale=1./255)
    train_data = datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='categorical')
    val_data = datagen.flow_from_directory(val_dir, target_size=(150,150), batch_size=32, class_mode='categorical')

    model = create_intel_model()
    model.fit(train_data, epochs=10, validation_data=val_data)

    # ✅ Ensure saved_models directory exists before saving
    os.makedirs('saved_models', exist_ok=True)
    model.save(os.path.join('saved_models', 'intel_model.h5'))
    print(" Model saved successfully at saved_models/intel_model.h5")
