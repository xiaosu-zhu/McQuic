# Training McQuic with custom configs

To train new McQuic models, you need to prepare a config which follows schema [schema.json](./schema.json) (also, [schema.md](./schema.md)).

# Model definition

In config, the `model` section gives hyper parameter for defining a model. For example, `channel` is channel of conv layers. All params are defined as in `mcquic/modules/compressor.py`.

# Training parameters

The `train` section defines training batch-size, optimizer, learning rate, *etc.* The desired `optimizer`/`lr scheduler`/... are retrieved by key in corresponding registry. 

You could see all entries of registries by calling registry's `.summary()` method.

Specifying debug mode in `mcquic train` could also show these entries.

Currently, registries are:

- `OptimRegistry`
- ...
