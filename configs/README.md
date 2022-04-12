# Training McQuic with custom configs

To train new McQuic models, you need to prepare a config which follows schema [schema.json](./schema.json) (also, [schema.md](./schema.md)).

# Model definition

In config, the `model` section gives hyper parameter for defining a model. For example, `channel` is channel of conv layers. All params are defined as in `mcquic/modules/compressor.py`.

# Training parameters

The `train` section defines training batch-size, optimizer, learning rate, *etc.* The desired `optimizer`/`lr scheduler`/... are retrieved by key in corresponding registry.

You could see all entries of registries by calling registry's `.summary()` method.

Specifying debug mode in `mcquic train` could also show these entries.

Currently, registries are:

- `OptimizerRegistry` contains Adam and LAMB optimizer currently.
- `LrSchedulerRegistry` contains most of learning rate schedulers in `torch.optim.lr_scheduler` as well as `mcquic.train.lrSchedulers`.
- `HookRegistry` contains trainer hooks.

For example, when
```yaml
...
train:
    ...
    optim:
        key: Lamb
        params:
            lr: 1.e-3
```
Then, "Lamb" optim will be retrieved from `OptimizerRegistry` and initialized by
## Extending registries by externalLibs

The `externalLib` in `train` provides interface to add external registry entries for extra training favors.

To acheive this, you should first write a python script that define some extra functions and register them with a registry. For example, if you have wrotten an amazing optimizer:
```python
# my_amaz_optim.py
...
class MyAmazOptim(torch.optim.Optimizer):
    pass
    ...
```
Then, you need to import `OptimRegistry` to register it.
```python
# my_amaz_optim.py
...
from mcquic.utils.registry import OptimRegistry

# or `@OptimRegistry.register("blabla")` to register with a specific key.
@OptimRegistry.register
class MyAmazOptim(torch.optim.Optimizer):
    pass
    ...
```
Next, how to ensure above code is visible to mcquic trainer? You only need to declare this file as an "external lib" in config:
```yaml
...
train:
    ...
    externalLib:
        - path/to/my_amaz_optim.py
...
```
You just need to give path of this file, and mcquic trainer will import it to register things in this file.

<a href="#">
  <image src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>
</a>

> The imported module is named by the MD5 hash of its absolute file path.

> To see which module is registered, you could enable `-D` flag in `mcquic train`. And all registries' summaries will be printed (mcquic/train/ddp.py:L98).

## Hooks called during training

When trainer is running, a bunch of hooks are called periodically. These are:

- BeforeRunHook
- AfterRunHook
- EpochStartHook
- EpochFinishHook
- StepStartHook
- StepFinishHook

Just named by when they are called.

To implement a hook, please refer to `mcquic/train/hooks.py`.

When hooks (registered key) are declared in
