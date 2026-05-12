# Model Configuration Guide
a complete overview of all configurable components in the system

This guide explains every part of the model that can be configured, what each configuration does, and where to find the available options. It is designed to help users understand how to modify the model's behaviour, architecture, training dynamics, and procedural input basis.

## 1. `config.py` — Core Model, Architecture & Training Configuration
`config.py` is the central configuration file for controlling the model's behaviour, architecture, training loop, generalisation settings, and viewer/telemetry options.

---

### 1.1 Architecture Configuration

#### `HIDDEN_LAYER_TOPOLOGY`

A list of integers defining the widths of the hidden layers.
<br>
Example:

`HIDDEN_LAYER_TOPOLOGY = [1280, 768, 512, 384, 256]`

The system automatically constructs the full topology as:

`[input feature count] + HIDDEN_LAYER_TOPOLOGY + [3]`

Users only modify the hidden layers.

#### `HIDDEN_ACT`

Selects the activation function used in all hidden layers.
<br>
All available activation functions  are defined in `backend_cupy.py` inside the `_ACT_MAP` dictionary.

__Important:__
<br>
Activation names ending in `_255` (e.g., `sigmoid_255`, `tanh_255`) are __output-scaled activations__ intended __only for the output layer.__
<br>
They should __not__ be used as hidden activations.

#### `OUTPUT_ACT`

Activation function for the output layer (RGB).
<br>
Options also come from `_ACT_MAP` within `backend_cupy.py`.

__Important:__
<br>
Activation names ending in `_255` (e.g., `sigmoid_255`, `tanh_255`) are __specifically designed for the output layer.__
<br>
These functions map values into the 0-255 pixel range and should be used for `OUTPUT_ACT`.
<br>
They are not suitable for hidden layers.

#### `PATCH_SIZE`

Patch size used for extracting local neighbourhoods.
<br>
Must be an odd integer.
<br>
Example:

`PATCH_SIZE = 7`

---

### 1.2 Training Behaviour


#### `EPOCHS`

Controls how many epochs the program will run from the moment you start it, not the total number of epochs the model must reach.

If you load an existing model that has already trained for 100 epochs, and you set:

`EPOCHS = 1000`

Then training will stop once the model reaches:

`100 (already trained) + 1000 (new session) = 1100 total epochs`

This makes `EPOCHS` a __relative value, not an absolute target__.


#### `TRAIN`

Enable or disable training mode.

#### `BATCH_SIZE`

Number of pixel-patch samples per batch.


#### `ENABLE_SHUFFLE`

Toggle shuffling pixel indices each epoch.

#### `LEARNING_RATE`

Initial learning rate.

#### `ENABLE_SET_LR`

Force the learning rate to the configured value when loading the model.
<br>
This is applied __once at load time__, not every epoch.

#### `ENABLE_ADAPTIVE_LR`

Enable adaptive learning rate logic.

#### `LR_INCREASE_MULTIPLIER` __/__ `LR_DECREASE_MULTIPLIER`

Controls how aggressively the adaptive learning rate adjusts.

#### `ADAPTIVE_LR_INVERTED`

Invert the adaptive learning rate behaviour. (e.g., when it would normally increase LR decrease it instead, when it would normally decrease LR increase it instead.)

#### `GRAD_CLIP`

Gradient clipping threshold.

#### `OPTIMISER`

Selects the optimiser from `optimiser_registry.py`.
<br>
Example:
<br>
`OPTIMISER = { "name": "rmsprop" }`

This also includes configurations for the optimiser.
<br>
Example:
<br>
`OPTIMISER = { "name": "rmsprop", "weight_decay": 0.01 }`

#### `LOSS_NAME`

Name of the loss function to use.

When set to `"wrapped_combined"`, `LOSS_CONFIG` is used to determine what loss functions are being combined.

#### `LOSS_CONFIG`

List of individual loss functions and their weighting.
<br>
Example:
<br>
`LOSS_CONFIG = [ ("mse", 1.0), ("mae", 1.0) ]`

Available loss functions are defined in `loss_registry.py`, under the `LOSS_REGISTRY` dictionary.

__Important:__

`"wrapped_combined"` is __only__ to be used for `LOSS_NAME` __never__ in `LOSS_CONFIG`. 
<br>
`"combined"` should not be used based on current architecture.
<br>
`LOSS_CONFIG` is only ever used when `LOSS_NAME` is set to `"wrapped_combined"`


#### `ENABLE_ADAPTIVE_LOSS_WEIGHTING`

Enable dynamic loss weighting.

#### `LOSS_WEIGHTING_POWER_SCALE`

Controls the strength of adaptive weighting.

---

### 1.3 Generalisation Settings

#### `ENABLE_ROTATE_TARGET_IMAGE`

Randomly rotate the target image during training.

#### `ROTATE_TARGET_FREQ`

How often to rotate (in epochs).

#### `MULTI_IMAGE_COUNT`

Number of images trained on simultaneously per epoch.

#### `HELDOUT_SEED`

Seed used for procedural input seeding for live image display.

Also used to determine which target image is shown in the live display when `ENABLE_CUSTOM_RESOLUTION == False`.

When `ENABLE_ROTATE_TARGET_IMAGE == False` and `MULTI_IMAGE_COUNT == 1`, this seed selects which entry in `Config/image_registry.json` is used for training.

__NOTE:__ If `Config/image_registry.json` does not exist, it will be created automatically when running `main.py`.

---

### 1.4 Viewer & Telemetry

#### `ENABLE_LIVE_VIEWER`

Enables generation of live-display images during training.

__NOTE:__ This does not launch the viewer; it only enables the model to produce images for the telemetry system

#### `ENABLE_END_VIEWER`

Shows the final viewer after training.
<br>
This viewer displays a frozen epoch, including the output image and all procedural inputs.
<br>
Use the arrow keys to navigate between images.

#### `ENABLE_TELEMETRY_VIEWER`

Reserved for future use; currently has no effect.

#### `LIVE_UPDATE_INTERVAL`

How often (in epochs) to generate the live-display image.

---

### 1.5 Image & Display Settings

#### `HEIGHT` __/__ `WIDTH`

Resolution of the live display image.

__NOTE:__ This does not affect training resolution; it only controls the resolution used for live display output.

#### `ENABLE_CUSTOM_RESOLUTION`

- When `True`:
<br>
The live display resolution is always taken from `HEIGHT` and `WIDTH`.

- When `False`:
<br>
The live display resolution is taken from the target image associated with `HELDOUT_SEED`.
<br>
If the seed does not map to any entry in `Config/image_registry.json`, the system falls back to `HEIGHT` and `WIDTH`.

#### `INPUT_CONFIG_PATH`

__Important:__
<br>
Do __not__ alter this value it is used to locate where the file is that tells the model what procedural inputs it uses.

---

### 1.6 Model Save/Load

#### `MODEL_SEED`

Seed for deterministic weight initialisation.

#### `FORCE_NEW_MODEL`

Force creation of new model file.
<br>
This will cause the model save file to be deleted.

#### `DEFAULT_MODEL_NAME`

Default name for saved models.
<br>
This is used for the file name.

#### `SAVE_FOLDER`

Directory where models are saved.

#### `SAVE_INTERVAL`

Save model every N epochs.


## 2. `pattern_config.py` — Procedural Input Basis

This file defines the __procedural input fields__ used by the model.
<br>
Each entry in `pattern_config` corresponds to one procedural field.

Example entry:

`{"type": "perlin", "frequency": 100.0, "octaves": 6}`

---

### 2.1 What Each Entry Contains

Each dictionary includes:

- `type` — The procedural generator name
- `seed` (optional) but also not something that should be manually used

There are many other options but you will need to check the specific procedural function to see their modification options as they vary drastically from one function to another.

These procedural functions can be found in the scripts within `Inputs/`

---

### 2.2 How The System Uses This File

- The list order defines the ordering of the fields.
- Each entry becomes a full image and patches are extracted from that image.
- Parameters define the shape and behaviour of each procedural field.
- Seeds are used for deterministic procedural fields and are used within the architecture and should never be manually set. Manually setting them will not break anything but should be avoided because its pointless and gets overwritten.

## 3. `cooling.py` — Thermal & Runtime Cooling Behaviour

This file controls GPU temperature management, cooling behaviour, and fan ramping.

---

### 3.1 Temperature Thresholds

- `MAX_TEMP` — The temp at which the entire main process will kill itself to prevent overheating
<br><br>

- `TARGET_TEMP_POST_EPOCH`
- `TARGET_TEMP_POST_BATCH`
- `TARGET_TEMP_PRE_DISPLAY`
- `TARGET_TEMP_DISPLAY_BATCH`

These determine what temperature the GPU will wait till it gets down to at each stage during training.

---

### 3.2 Cooling Toggles

__NOTE:__ All batch cooling has a toggle for a shallow version of its cooling it changes the method of cooling from a target GPU temp to a specified timeframe (seconds).

- `ENABLE_BATCH_COOLING`
- `ENABLE_SHALLOW_BATCH_COOLING`
<br>

For this cooling logic to be used `ENABLE_BATCH_COOLING` must be set to `True`. But when `ENABLE_SHALLOW_BATCH_COOLING` is `True` it changes how the cooling logic works from a target GPU temp to a specified interval.

- `ENABLE_PRE_DISPLAY_COOLING` — Occurs after training for the epoch but before generating the image for live display
<br>
<br>
- `ENABLE_DISPLAY_BATCH_COOLING`
- `ENABLE_SHALLOW_DISPLAY_BATCH_COOLING`
<br>

This is just like with `ENABLE_BATCH_COOLING` and `ENABLE_SHALLOW_BATCH_COOLING` it just happens after each batch when generating the live image rather than after each batch during training. 

---

### 3.3 Shallow Batch Cooling

- `SHALLOW_BATCH_COOL_TIME`

Specifies the time (seconds) that the process will sleep for after each batch during training if `SHALLOW_BATCH_COOLING = True` and `ENABLE_BATCH_COOLING = True`

Example:
<br>
`SHALLOW_BATCH_COOL_TIME = 0.05`

- `SHALLOW_DISPLAY_COOL_TIME`

---

### 3.4 Fan Behaviour

- `FAN_RAMP_START`
- `MAX_SAFE_FAN`
- `FAN_RAMP`

__NOTE:__ Fan ramp behaviour has not been properly implemented for windows and the method currently implemented only works on linux but has not been tested so this is not currently relevant.

---

### 3.5 VRAM Behaviour

- `PROFILE_VRAM`
- `VRAM_HEADROOM`

__Note:__ These variables are not used at all so are not relevant.

## 4. Reference Files/Folders (Not Modified, But Important)

These files define the available options for the configs above.

---

### 4.1 `pattern_registry.py` __/__ `src/inputs/` — Procedural Pattern Definitions

This file contains all available procedural field types.

Refer to the files within the folder to see:

- Specific constructors
- Configuration options
- How they generate their fields

---

### 4.2 `optimiser_registry.py` — Available Optimisers

Defines all optimiser names that can be using in:

`OPTIMISER = { "name": "..." }`

Examples include:

- `"sgd"`
- `"nesterov"`
- `"lion"`
- `"adabelief"`

---

### 4.3 `loss_registry.py` — Available Loss Functions

Defines all loss functions under `LOSS_REGISTRY` dictionary, that can be used in:

`LOSS_CONFIG = [ ("loss_name", weight) ]` 

Examples:

- `"mse"`
- `"mae"`

---

### 4.4 `backend_cupy.py` — Activation Functions

Contains all activation functions available for:

- `HIDDEN_ACT`
- `OUTPUT_ACT`

Examples:

- `sin`
- `cos`
- `relu`
- `tanh`
- `tanh_255`
- `sigmoid_255`

