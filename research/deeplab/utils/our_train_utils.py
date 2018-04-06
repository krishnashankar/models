"""Utility functions for training."""

import tensorflow as tf

slim = tf.contrib.slim


def add_softmax_cross_entropy_loss(logits,
                                   labels,
                                   num_classes,
                                   ignore_label,
                                   loss_weight=1.0,
                                   scope=None):
  """Adds softmax cross entropy loss for logits of each scale.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    loss_weight: Float, loss weight.
    upsample_logits: Boolean, upsample logits or not.
    scope: String, the scope for the loss.

  Raises:
    ValueError: Label or logits is None.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  scaled_labels = tf.reshape(labels, shape=[-1])
  not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                             ignore_label)) * loss_weight
  one_hot_labels = slim.one_hot_encoding(
      scaled_labels, num_classes, on_value=1.0, off_value=0.0)
  tf.losses.softmax_cross_entropy(
      one_hot_labels,
      tf.reshape(logits, shape=[-1, num_classes]),
      weights=not_ignore_mask,
      scope=scope)


def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
  """Returns the subset of variables available in the checkpoint.

  Inspects given checkpoint and returns the subset of variables that are
  available in it.

  TODO(rathodv): force input and output to be a dictionary.

  Args:
    variables: a list or dictionary of variables to find in checkpoint.
    checkpoint_path: path to the checkpoint to restore variables from.
    include_global_step: whether to include `global_step` variable, if it
      exists. Default True.

  Returns:
    A list or dictionary of variables.
  Raises:
    ValueError: if `variables` is not a list or dict.
  """
  if isinstance(variables, list):
    variable_names_map = {variable.op.name: variable for variable in variables}
  elif isinstance(variables, dict):
    variable_names_map = variables
  else:
    raise ValueError('`variables` is expected to be a list or dict.')
  ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
  ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
  if not include_global_step:
    ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
  vars_in_ckpt = {}
  for variable_name, variable in sorted(variable_names_map.items()):
    if variable_name in ckpt_vars_to_shape_map:
      if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
        vars_in_ckpt[variable_name] = variable
      else:
        tf.logging.warning('Variable [%s] is available in checkpoint, but has an '
                        'incompatible shape with model variable.',
                        variable_name)
    else:
      tf.logging.warning('Variable [%s] is not available in checkpoint',
                      variable_name)
  if isinstance(variables, list):
    return vars_in_ckpt.values()
  print(vars_in_ckpt)
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  return vars_in_ckpt


  
def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
  if tf_initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization; other checkpoint exists')
    return None

  tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
  exclude_list = ['global_step']
  variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)

  available_var_map = (
      get_variables_available_in_checkpoint(
          variables_to_restore, tf_initial_checkpoint,
          include_global_step=False))

  
  return slim.assign_from_checkpoint_fn(
      tf_initial_checkpoint,
      available_var_map,
      ignore_missing_vars=ignore_missing_vars)


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.

  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.

  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.

  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
  gradient_multipliers = {}

  for var in slim.get_model_variables():
    # Double the learning rate for biases.
    if 'biases' in var.op.name:
      gradient_multipliers[var.op.name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in var.op.name and 'biases' in var.op.name:
        gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in var.op.name:
        gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers


def get_model_learning_rate(
    learning_policy, base_learning_rate, learning_rate_decay_step,
    learning_rate_decay_factor, training_number_of_steps, learning_power,
    slow_start_step, slow_start_learning_rate):
  """Gets model's learning rate.

  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.

  Returns:
    Learning rate for the specified learning policy.

  Raises:
    ValueError: If learning policy is not recognized.
  """
  global_step = tf.train.get_or_create_global_step()
  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        global_step,
        training_number_of_steps,
        end_learning_rate=0,
        power=learning_power)
  else:
    raise ValueError('Unknown learning policy.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                  learning_rate)
