from keras.layers import Input, BatchNormalization, Conv2D, Activation, Add, Flatten, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
MODEL_FILTERS = 128
N_BLOCKS = 8
LR = 1e-3
REG_CONST = 1e-4

def resnet_block(x, filters):

    shortcut = x

    x = Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=l2(REG_CONST))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=l2(REG_CONST))(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_chessnet(model_filters=MODEL_FILTERS, n_blocks=N_BLOCKS, lr=LR):
    inputs = Input(shape=(8,8,6))
    x = Conv2D(MODEL_FILTERS, 3, padding='same', use_bias=False, kernel_regularizer=l2(REG_CONST))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for _ in range(n_blocks):
        x = resnet_block(x, model_filters)

    # Policy Head
    policy = Conv2D(32, 1, padding='same', use_bias=False, kernel_regularizer=l2(REG_CONST))(x)
    policy = BatchNormalization()(policy)
    policy = Activation('relu')(policy)
    policy = Conv2D(56, 1, padding='same', use_bias=True, kernel_regularizer=l2(REG_CONST))(policy)
    policy_out = Flatten(name='policy_output')(policy)

    # Value Head

    value = Conv2D(32, 1, padding='same', use_bias=False, kernel_regularizer=l2(REG_CONST))(x)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = Flatten()(value)
    value = Dense(128, activation='relu', kernel_regularizer=l2(REG_CONST))(value)
    value_out = Dense(3, activation='softmax', name='value_output', kernel_regularizer=l2(REG_CONST))(value)

    chessnet = Model(inputs=inputs, outputs=[policy_out, value_out])
    chessnet.compile(
            optimizer=Adam(learning_rate=LR),
            loss={
                'policy_output': CategoricalCrossentropy(from_logits=True),
                'value_output': CategoricalCrossentropy(from_logits=False)
                },
            loss_weights={
                'policy_output': 2.0,
                'value_output': 1.0,
                })
    return chessnet
