import numpy as np

def make_erp(
    n_epochs = 100,
    n_channels = 64,
    length = 1, # in second
    fs = 250,
    ratio_target = 0.2,
    random_state = None,
    erp_range = [0.2, 0.5],
    normal_scale = 1.0,
    erp_scale = 1.0
):

  rng = np.random.default_rng(seed=random_state)
  data = rng.normal(loc=0.0, scale=normal_scale, size=(n_epochs, n_channels, fs*length))

  y = np.zeros(100, dtype = np.int64)

  n_target = int(n_epochs * ratio_target)
  y[0:n_target] = np.ones(n_target)

  rng.shuffle(y)
  erp_range_sample = np.array(erp_range)*fs
  erp_range_sample = erp_range_sample.astype(np.int64)

  for idx, val in enumerate(y):
    if val == 1:
      data[idx, :, erp_range_sample[0]:erp_range_sample[1]] = data[idx, :, erp_range_sample[0]:erp_range_sample[1]] + 1.0*erp_scale

  return data