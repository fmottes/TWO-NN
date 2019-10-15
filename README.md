# TWO-NN

Naive Python 3 implementation of TWO-NN algorithm for intrinsic dimension inference.

Dependencies
---
* Python >= 3.6
* Numpy >= 1.17

Usage
---
```python
import numpy as np
from TwoNN import twonn_dimension

#mock dataset - 1000 samples with 500 features
data = np.random.uniform(0,1,size=(1000,500))

#calculate intrinsic dimension d
d = twonn_dimension(data)
```

References
---
E. Facco, M. d’Errico, A. Rodriguez & A. Laio, Estimating the intrinsic dimension of datasets by a minimal neighborhood information, *Scientific Reports*, 2017

(https://doi.org/10.1038/s41598-017-11873-y)
    
