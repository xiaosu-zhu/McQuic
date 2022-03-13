# Config schema

- [1. ![badge](https://img.shields.io/badge/Optional-yellow) Property `model`](#model)
  - [1.1. Property `None`](#model_allOf_i0)
- [2. ![badge](https://img.shields.io/badge/Optional-yellow) Property `train`](#train)
  - [2.1. Property `None`](#train_allOf_i0)
    - [2.1.1. Property `trainSet`](#train_allOf_i0_trainSet)
    - [2.1.2. Property `valFreq`](#train_allOf_i0_valFreq)
    - [2.1.3. Property `valSet`](#train_allOf_i0_valSet)
    - [2.1.4. Property `optim`](#train_allOf_i0_optim)
      - [2.1.4.1. Property `None`](#train_allOf_i0_optim_allOf_i0)
    - [2.1.5. Property `saveDir`](#train_allOf_i0_saveDir)
    - [2.1.6. Property `target`](#train_allOf_i0_target)
    - [2.1.7. Property `schdr`](#train_allOf_i0_schdr)
      - [2.1.7.1. Property `None`](#train_allOf_i0_schdr_allOf_i0)
    - [2.1.8. Property `batchSize`](#train_allOf_i0_batchSize)
    - [2.1.9. Property `gpu`](#train_allOf_i0_gpu)
      - [2.1.9.1. Property `None`](#train_allOf_i0_gpu_allOf_i0)
    - [2.1.10. Property `epoch`](#train_allOf_i0_epoch)

**Title:** Config schema

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** The bravo schema for writing a config!

| Property           | Pattern | Type        | Deprecated | Definition | Title/Description                                                                    |
| ------------------ | ------- | ----------- | ---------- | ---------- | ------------------------------------------------------------------------------------ |
| - [model](#model ) | No      | Combination | No         | -          | Compression model to use. Now we only have one model, so 'key' is ignored. Avali ... |
| - [train](#train ) | No      | Combination | No         | -          | Training configs.                                                                    |
|                    |         |             |            |            |                                                                                      |

## <a name="model"></a>1. ![badge](https://img.shields.io/badge/Optional-yellow) Property `model`

| Type                      | `combining`                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Compression model to use. Now we only have one model, so `key` is ignored. Avaliable params are `channel`, `m` and `k`.

| All of(Requirement)        |
| -------------------------- |
| [General](#model_allOf_i0) |
|                            |

### <a name="model_allOf_i0"></a>1.1. Property `None`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Same definition as**    | [Train_optim_allOf_i0](#Train_optim_allOf_i0)                                                                       |
|                           |                                                                                                                     |

## <a name="train"></a>2. ![badge](https://img.shields.io/badge/Optional-yellow) Property `train`

| Type                      | `combining`                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Training configs.

| All of(Requirement)      |
| ------------------------ |
| [Train](#train_allOf_i0) |
|                          |

### <a name="train_allOf_i0"></a>2.1. Property `None`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/Train                                                                                                             |
|                           |                                                                                                                     |

| Property                                  | Pattern | Type             | Deprecated | Definition | Title/Description                                                                    |
| ----------------------------------------- | ------- | ---------------- | ---------- | ---------- | ------------------------------------------------------------------------------------ |
| - [trainSet](#train_allOf_i0_trainSet )   | No      | string           | No         | -          | A dir path to load 'lmdb' dataset. You need to convert your images before you gi ... |
| - [valFreq](#train_allOf_i0_valFreq )     | No      | integer          | No         | -          | Run validation after every 'valFreq' epochs.                                         |
| - [valSet](#train_allOf_i0_valSet )       | No      | string           | No         | -          | A dir path to load image files for validation.                                       |
| - [optim](#train_allOf_i0_optim )         | No      | Combination      | No         | -          | Optimizer used for training. As for current we have 'Adam' and 'Lamb'.               |
| - [saveDir](#train_allOf_i0_saveDir )     | No      | string           | No         | -          | A dir path to save model checkpoints, TensorBoard messages and logs.                 |
| - [target](#train_allOf_i0_target )       | No      | enum (of string) | No         | -          | Training target. Now is one of '[PSNR, MsSSIM]'.                                     |
| - [schdr](#train_allOf_i0_schdr )         | No      | Combination      | No         | -          | Learning rate scheduler used for training. As for current we have 'ReduceLROnPla ... |
| - [batchSize](#train_allOf_i0_batchSize ) | No      | integer          | No         | -          | Batch size for training. NOTE: The actual batch size (whole world) is computed b ... |
| - [gpu](#train_allOf_i0_gpu )             | No      | Combination      | No         | -          | GPU configs for training.                                                            |
| - [epoch](#train_allOf_i0_epoch )         | No      | integer          | No         | -          | Total training epochs.                                                               |
|                                           |         |                  |            |            |                                                                                      |

#### <a name="train_allOf_i0_trainSet"></a>2.1.1. Property `trainSet`

| Type                      | `string`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** A dir path to load `lmdb` dataset. You need to convert your images before you give this path by calling `mcquic dataset ...`.

#### <a name="train_allOf_i0_valFreq"></a>2.1.2. Property `valFreq`

| Type                      | `integer`                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Run validation after every `valFreq` epochs.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |
|              |        |

#### <a name="train_allOf_i0_valSet"></a>2.1.3. Property `valSet`

| Type                      | `string`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** A dir path to load image files for validation.

#### <a name="train_allOf_i0_optim"></a>2.1.4. Property `optim`

| Type                      | `combining`                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Optimizer used for training. As for current we have `Adam` and `Lamb`.

| All of(Requirement)                       |
| ----------------------------------------- |
| [General](#train_allOf_i0_optim_allOf_i0) |
|                                           |

##### <a name="train_allOf_i0_optim_allOf_i0"></a>2.1.4.1. Property `None`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Same definition as**    | [Train_optim_allOf_i0](#Train_optim_allOf_i0)                                                                       |
|                           |                                                                                                                     |

#### <a name="train_allOf_i0_saveDir"></a>2.1.5. Property `saveDir`

| Type                      | `string`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** A dir path to save model checkpoints, TensorBoard messages and logs.

#### <a name="train_allOf_i0_target"></a>2.1.6. Property `target`

| Type                      | `enum (of string)`                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Training target. Now is one of `[PSNR, MsSSIM]`.

Must be one of:
* "PSNR"
* "MsSSIM"

#### <a name="train_allOf_i0_schdr"></a>2.1.7. Property `schdr`

| Type                      | `combining`                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Learning rate scheduler used for training. As for current we have `ReduceLROnPlateau`, `Exponential`, `MultiStep`, `OneCycle` and all schedulers defined in `mcquic.train.lrSchedulers`.

| All of(Requirement)                       |
| ----------------------------------------- |
| [General](#train_allOf_i0_schdr_allOf_i0) |
|                                           |

##### <a name="train_allOf_i0_schdr_allOf_i0"></a>2.1.7.1. Property `None`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Same definition as**    | [Train_optim_allOf_i0](#Train_optim_allOf_i0)                                                                       |
|                           |                                                                                                                     |

#### <a name="train_allOf_i0_batchSize"></a>2.1.8. Property `batchSize`

| Type                      | `integer`                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Batch size for training. NOTE: The actual batch size (whole world) is computed by `batchSize * gpus`.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |
|              |        |

#### <a name="train_allOf_i0_gpu"></a>2.1.9. Property `gpu`

| Type                      | `combining`                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** GPU configs for training.

| All of(Requirement)                 |
| ----------------------------------- |
| [GPU](#train_allOf_i0_gpu_allOf_i0) |
|                                     |

##### <a name="train_allOf_i0_gpu_allOf_i0"></a>2.1.9.1. Property `None`

| Type                      | `object`                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
| **Same definition as**    | [Train_gpu_allOf_i0](#Train_gpu_allOf_i0)                                                                           |
|                           |                                                                                                                     |

#### <a name="train_allOf_i0_epoch"></a>2.1.10. Property `epoch`

| Type                      | `integer`                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Additional properties** | [![badge](https://img.shields.io/badge/Any+type-allowed-green)](# "Additional Properties of any type are allowed.") |
|                           |                                                                                                                     |

**Description:** Total training epochs.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |
|              |        |

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans)