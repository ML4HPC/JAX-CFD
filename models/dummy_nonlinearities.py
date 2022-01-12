import gin


def identity(x):
  return x


identity = gin.external_configurable(identity)
