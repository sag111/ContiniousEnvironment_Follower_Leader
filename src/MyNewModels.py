import numpy as np
import gym
from typing import Dict, Optional, Sequence
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType, List, ModelConfigDict
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.tf_utils import one_hot

tf1, tf, tfv = try_import_tf()


class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(self.inputs)

        layer_2 = tf.keras.layers.Dense(
            256,
            name="my_layer2",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(layer_1)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_2)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_2)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}


class MyKerasModelPrev(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasModelPrev, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = self.inputs

        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        layer_1 = tf.keras.layers.Dense(
            128,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(x)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}


class MyLSTMModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyLSTMModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = self.inputs

        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        layer_1 = tf.keras.layers.Dense(
            128,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(x)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}


class DefaultModelPrev(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DefaultModelPrev, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = self.inputs

        x = tf.keras.layers.Flatten(data_format="channels_first")(x)

        x = tf.keras.layers.Dense(
            256,
            name="my_layer2",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(x)

        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(x)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    # def metrics(self):
    #     return {"foo": tf.constant(42.0)}
    #


class DefaultLSTMModelPrev(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DefaultLSTMModelPrev, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = self.inputs

        # x = tf.keras.layers.Flatten(data_format="channels_first")(x)
        x = tf.keras.layers.LSTM(256, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(128, return_sequences=False)(x)

        x = tf.keras.layers.Dense(
            256,
            name="my_layer2",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(x)

        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(x)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    # def metrics(self):
    #     return {"foo": tf.constant(42.0)}
    #


class DefaultModelPrev_V2(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DefaultModelPrev_V2, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = self.inputs

        x = tf.keras.layers.Flatten(data_format="channels_first")(x)

        x = tf.keras.layers.Dense(
            512,
            name="my_layer3",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(x)
        x = tf.keras.layers.Dense(
            512,
            name="my_layer2",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(x)

        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(x)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    # def metrics(self):
    #     return {"foo": tf.constant(42.0)}
    #


class BiLSTMModelPrevV2(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(BiLSTMModelPrevV2, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = self.inputs

        # x = tf.keras.layers.Flatten(data_format="channels_first")(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.1))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.1))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.1))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))(x)


        x = tf.keras.layers.Dense(
            256,
            name="my_layer2",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(x)

        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(x)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


# ModelCatalog.register_custom_model("default_keras_model", MyKerasModel)

# ModelCatalog.register_custom_model("default_keras_model_prev", MyKerasModelPrev)

ModelCatalog.register_custom_model("def_m_prev", DefaultModelPrev)

ModelCatalog.register_custom_model("def_m_prev_v2", DefaultModelPrev_V2)

ModelCatalog.register_custom_model("def_lstm_m_prev", DefaultLSTMModelPrev)

ModelCatalog.register_custom_model("bi_lstm_v2_m_prev", BiLSTMModelPrevV2)