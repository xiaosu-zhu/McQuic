# Training McQuic with custom configs

To train new McQuic models, you need to prepare a config which follows schema [schema.json](./schema.json) (also, [schema.md](./schema.md)).

# Model definition

In config, the `model` section gives hyper parameter for defining a model. For example, `channel` is channel of conv layers. All params are defined as in `mcquic/modules/compressor.py`.

# Training parameters

The `train` section declares training batch-size, optimizer, learning rate, *etc.* The desired `optimizer`/`lr scheduler`/... are retrieved by key in corresponding registry.

You could see all entries of registries by calling registry's `.summary()` method.

Specifying debug `-D` flag in `mcquic train` could also show these entries.

Currently, registries are:

- `OptimizerRegistry` has some torch optimizers.
- `LrSchedulerRegistry` has some learning rate schedulers from `torch.optim.lr_scheduler` and `mcquic.train.lrSchedulers`.
- `LossRegistry` has distortion (and may also has rate) objectives.
- `HookRegistry` has hooks for trainer.

For example, following config:
```yaml
...
train:
  optim:
    key: Lamb
    params:
      lr: 1.e-3
```
will get LAMB optimizer from `OptimizerRegistry` by key `Lamb`, and initialize it by params as follows:
```python
optimClass = OptimizerRegistry.get(config.Train.Optim.Key) # "Lamb"
optim = optimClass(**config.Train.Optim.Params) # equivalent to `Lamb(lr=1e-3)`
```

## Extending registry by external libs

`externalLib` in `train` section provides interface to include python scripts defined outside mcquic. Therefore, you could register your custom functions to extend training components. For example, if you have built an amazing optimzer:
```python
# my_amaz_optim.py
...

class MyAmazOptim(torch.optim.Optimizer):
  ...
```
The way to use it for mcquic training is to import `OptimizerRegistry` and register your optimizer into it:
```python
# my_amaz_optim.py
...
from mcquic.utils.registry import OptimizerRegistry

# or @OptimizerRegistry.register("amaz") to give a special name
@OptimizerRegistry.register
class MyAmazOptim(torch.optim.Optimizer):
  ...
```

Then, in config, you could declare your python file by path in `externalLib` part:
```yaml
...
train:
  externalLib:
    - path/to/my_amaz_optim.py
    # - other/modules/to/be/registered.py
  optim:
    key: MyAmazOptim
    params:
      myparam: myvalue
```
And `mcquic train` will import this file to make your code be visible to it. Then, `MyAmazOptim` will be retrieved and initialized by `MyAmazOptim(myparam=myvalue)`.

<a href="#">
  <image src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>
</a>

> The imported module will be named as its MD5 hash of the absolute file path.

> To check all registered things, you could specify `-D` flag in `mcquic train`. And registry information will be logged as in `mcquic/train/ddp.py#L97`.

## Hooks in trainer

Hooks are called periodically during training. As their names show, they are called when train starts/stops, epoch begins/ends and single step starts/ends, all hooks are:
```
BeforeRunHook
AfterRunHook
EpochStartHook
EpochFinishHook
StepStartHook
StepFinishHook
```
To implement hooks, please refer to `mcquic/train/hooks.py`.

### Use hooks
`hooks` in `train` section declare hooks should be installed in this run. By using [external libs](#extending-registry-by-external-libs), you could insert your own hooks. Then:
```yaml
...
train:
  externalLib:
    - path/to/my_amaz_hook.py # where you defined `MyAmazHook`
  hooks:
    - key: MyAmazHook
      params:
        myparam: myvalue
```
Similiary, your amazing hook will be retrieved and initialized by `MyAmazHook(myparam=myvalue)` and installed to trainer (depeding on which kinds of hooks you've implemented).
