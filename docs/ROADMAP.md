1. Include a module with all optimizers from docs/OPTIMIZERS.md into src/model/optimizers.py
2. Include in schemas a optimizer chooser, with name and specific hiperparameters for each optimizer, with a prefix on the name of the chosed optimizer.
For example a tentative approach would be something similar to:
```yaml
model class:
model:
  ....
training:
  ...
  optimizer:
    optimizer_class: Adam
    parameters:
      adam-momentum: ...
      ...
```
