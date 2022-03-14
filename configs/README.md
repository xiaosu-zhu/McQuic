# Schema Docs

- [1. ![badge](https://img.shields.io/badge/Required-blue) Property `model`](#model)
  - [1.1. ![badge](https://img.shields.io/badge/Required-blue) Property `key`](#model_key)
  - [1.2. ![badge](https://img.shields.io/badge/Required-blue) Property `params`](#model_params)
- [2. ![badge](https://img.shields.io/badge/Required-blue) Property `train`](#train)
  - [2.1. ![badge](https://img.shields.io/badge/Required-blue) Property `batchSize`](#train_batchSize)
  - [2.2. ![badge](https://img.shields.io/badge/Required-blue) Property `epoch`](#train_epoch)
  - [2.3. ![badge](https://img.shields.io/badge/Required-blue) Property `gpu`](#train_gpu)
    - [2.3.1. ![badge](https://img.shields.io/badge/Required-blue) Property `gpus`](#train_gpu_gpus)
    - [2.3.2. ![badge](https://img.shields.io/badge/Required-blue) Property `vRam`](#train_gpu_vRam)
    - [2.3.3. ![badge](https://img.shields.io/badge/Required-blue) Property `wantsMore`](#train_gpu_wantsMore)
  - [2.4. ![badge](https://img.shields.io/badge/Required-blue) Property `optim`](#train_optim)
  - [2.5. ![badge](https://img.shields.io/badge/Required-blue) Property `saveDir`](#train_saveDir)
  - [2.6. ![badge](https://img.shields.io/badge/Required-blue) Property `schdr`](#train_schdr)
  - [2.7. ![badge](https://img.shields.io/badge/Required-blue) Property `target`](#train_target)
  - [2.8. ![badge](https://img.shields.io/badge/Required-blue) Property `trainSet`](#train_trainSet)
  - [2.9. ![badge](https://img.shields.io/badge/Required-blue) Property `valFreq`](#train_valFreq)
  - [2.10. ![badge](https://img.shields.io/badge/Required-blue) Property `valSet`](#train_valSet)

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/definitions/ConfigSchema                                                                                          |
|                           |                                                                                                                     |

| Property           | Pattern | Type   | Deprecated | Definition                     | Title/Description                                                                    |
| ------------------ | ------- | ------ | ---------- | ------------------------------ | ------------------------------------------------------------------------------------ |
| + [model](#model ) | No      | object | No         | In #/definitions/GeneralSchema | Compression model to use. Now we only have one model, so 'key' is ignored. Avali ... |
| + [train](#train ) | No      | object | No         | In #/definitions/TrainSchema   | Training configs.                                                                    |
|                    |         |        |            |                                |                                                                                      |

## <a name="model"></a>1. ![badge](https://img.shields.io/badge/Required-blue) Property `model`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/definitions/GeneralSchema                                                                                         |
|                           |                                                                                                                     |

**Description:** Compression model to use. Now we only have one model, so `key` is ignored. Avaliable params are `channel`, `m` and `k`.

| Property                   | Pattern | Type                      | Deprecated | Definition | Title/Description |
| -------------------------- | ------- | ------------------------- | ---------- | ---------- | ----------------- |
| + [key](#model_key )       | No      | string                    | No         | -          | key               |
| + [params](#model_params ) | No      | string, number or boolean | No         | -          | params            |
|                            |         |                           |            |            |                   |

### <a name="model_key"></a>1.1. ![badge](https://img.shields.io/badge/Required-blue) Property `key`

**Title:** key

| Type                      | `string`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** A unique key used to retrieve in registry. For example, given `Lamb` for optimizers, it will check `OptimRegistry` and find the optimizer `apex.optim.FusedLAMB`.

### <a name="model_params"></a>1.2. ![badge](https://img.shields.io/badge/Required-blue) Property `params`

**Title:** params

| Type                      | `string, number or boolean`                                                                                                                                      |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Should-conform-blue)](#model_params_additionalProperties "Each additional property must conform to the following schema") |
|                           |                                                                                                                                                                  |

**Description:** Corresponding funcation call parameters. So the whole call is `registry.get(key)(**params)`.

## <a name="train"></a>2. ![badge](https://img.shields.io/badge/Required-blue) Property `train`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/definitions/TrainSchema                                                                                           |
|                           |                                                                                                                     |

**Description:** Training configs.

| Property                         | Pattern | Type             | Deprecated | Definition                 | Title/Description                                                                    |
| -------------------------------- | ------- | ---------------- | ---------- | -------------------------- | ------------------------------------------------------------------------------------ |
| + [batchSize](#train_batchSize ) | No      | integer          | No         | -                          | batchSize                                                                            |
| + [epoch](#train_epoch )         | No      | integer          | No         | -                          | epoch                                                                                |
| + [gpu](#train_gpu )             | No      | object           | No         | In #/definitions/GPUSchema | GPU configs for training.                                                            |
| + [optim](#train_optim )         | No      | object           | No         | Same as [model](#model )   | Optimizer used for training. As for current we have 'Adam' and 'Lamb'.               |
| + [saveDir](#train_saveDir )     | No      | string           | No         | -                          | saveDir                                                                              |
| + [schdr](#train_schdr )         | No      | object           | No         | Same as [model](#model )   | Learning rate scheduler used for training. As for current we have 'ReduceLROnPla ... |
| + [target](#train_target )       | No      | enum (of string) | No         | -                          | target                                                                               |
| + [trainSet](#train_trainSet )   | No      | string           | No         | -                          | trainSet                                                                             |
| + [valFreq](#train_valFreq )     | No      | integer          | No         | -                          | valFreq                                                                              |
| + [valSet](#train_valSet )       | No      | string           | No         | -                          | valSet                                                                               |
|                                  |         |                  |            |                            |                                                                                      |

### <a name="train_batchSize"></a>2.1. ![badge](https://img.shields.io/badge/Required-blue) Property `batchSize`

**Title:** batchSize

| Type                      | `integer`                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Batch size for training. NOTE: The actual batch size (whole world) is computed by `batchSize * gpus`.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |
|              |        |

### <a name="train_epoch"></a>2.2. ![badge](https://img.shields.io/badge/Required-blue) Property `epoch`

**Title:** epoch

| Type                      | `integer`                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Total training epochs.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |
|              |        |

### <a name="train_gpu"></a>2.3. ![badge](https://img.shields.io/badge/Required-blue) Property `gpu`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/definitions/GPUSchema                                                                                             |
|                           |                                                                                                                     |

**Description:** GPU configs for training.

| Property                             | Pattern | Type    | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ------- | ---------- | ---------- | ----------------- |
| + [gpus](#train_gpu_gpus )           | No      | integer | No         | -          | gpus              |
| + [vRam](#train_gpu_vRam )           | No      | integer | No         | -          | vRam              |
| + [wantsMore](#train_gpu_wantsMore ) | No      | boolean | No         | -          | wantsMore         |
|                                      |         |         |            |            |                   |

#### <a name="train_gpu_gpus"></a>2.3.1. ![badge](https://img.shields.io/badge/Required-blue) Property `gpus`

**Title:** gpus

| Type                      | `integer`                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Number of gpus for training. This affects the `world size` of PyTorch DDP.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |
|              |        |

#### <a name="train_gpu_vRam"></a>2.3.2. ![badge](https://img.shields.io/badge/Required-blue) Property `vRam`

**Title:** vRam

| Type                      | `integer`                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Minimum VRam required for each gpu. Set it to `-1` to use all gpus.

#### <a name="train_gpu_wantsMore"></a>2.3.3. ![badge](https://img.shields.io/badge/Required-blue) Property `wantsMore`

**Title:** wantsMore

| Type                      | `boolean`                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Set to `true` to use all visible gpus and all VRams and ignore `gpus` and `vRam`.

### <a name="train_optim"></a>2.4. ![badge](https://img.shields.io/badge/Required-blue) Property `optim`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Same definition as**    | [model](#model)                                                                                                     |
|                           |                                                                                                                     |

**Description:** Optimizer used for training. As for current we have `Adam` and `Lamb`.

### <a name="train_saveDir"></a>2.5. ![badge](https://img.shields.io/badge/Required-blue) Property `saveDir`

**Title:** saveDir

| Type                      | `string`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** A dir path to save model checkpoints, TensorBoard messages and logs.

### <a name="train_schdr"></a>2.6. ![badge](https://img.shields.io/badge/Required-blue) Property `schdr`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Same definition as**    | [model](#model)                                                                                                     |
|                           |                                                                                                                     |

**Description:** Learning rate scheduler used for training. As for current we have `ReduceLROnPlateau`, `Exponential`, `MultiStep`, `OneCycle` and all schedulers defined in `mcquic.train.lrSchedulers`.

### <a name="train_target"></a>2.7. ![badge](https://img.shields.io/badge/Required-blue) Property `target`

**Title:** target

| Type                      | `enum (of string)`                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Training target. Now is one of `[PSNR, MsSSIM]`.

Must be one of:
* "PSNR"
* "MsSSIM"

### <a name="train_trainSet"></a>2.8. ![badge](https://img.shields.io/badge/Required-blue) Property `trainSet`

**Title:** trainSet

| Type                      | `string`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** A dir path to load `lmdb` dataset. You need to convert your images before you give this path by calling `mcquic dataset ...`.

### <a name="train_valFreq"></a>2.9. ![badge](https://img.shields.io/badge/Required-blue) Property `valFreq`

**Title:** valFreq

| Type                      | `integer`                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Run validation after every `valFreq` epochs.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |
|              |        |

### <a name="train_valSet"></a>2.10. ![badge](https://img.shields.io/badge/Required-blue) Property `valSet`

**Title:** valSet

| Type                      | `string`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** A dir path to load image files for validation.

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans)