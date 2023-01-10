# Install Homebrew on m1
```cmd
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

# Install Miniforge for GPU acceleration support on m1
```cmd
brew install miniforge
```

# Install PyTorch Latest version 
```cmd
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

* run the following to check if MPS support is enabled in python
```python
import torch
torch.backends.mps.is_available()
```

# Install Tensorflow with GPU support 
```cmd
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```

* Tensorflow GPU Check
```python
import tensorflow as tf
GPU_info = tf.config.list_physical_devices('GPU')
print('GPU available: ", len(GPU_info)>0, 'GPU info - ', GPU_info)
```
