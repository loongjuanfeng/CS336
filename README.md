## CS336 assignment 1

The tokenizer is available from `tokenizer`.  The neural-network and training
utilities are intentionally small library modules:

```python
from model import TransformerLanguageModel
from optimization import AdamW
from training import train
from generation import generate
```

Construct a `TransformerLanguageModel`, an `AdamW` optimizer, and call
`training.train` with a one-dimensional NumPy token array (regular arrays and
`numpy.memmap` inputs are supported).  `generation.generate` accepts a
two-dimensional batch of token IDs and provides temperature, top-p, and EOS
sampling controls.
