# computer_vision_covid_and_any

Я взял набор данных из kaggle, в наборе данных было 15 тысяч обучающих и 5,3 тысячи тестовых данных, так что обучение было недолгим, я удалил несколько тысяч элементов. Тренинг длился примерно 2100 секунд, что является хорошей практикой для изучения CV
    
    model = models.Sequential([
    
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.MaxPool2D(strides=2),
    layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D(strides=2),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D(strides=2),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D(strides=2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(4, activation='softmax')
    ])
