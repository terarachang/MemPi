#### We store the tokenized data, which are retrieved from [batch_viewer.py](https://github.com/EleutherAI/pythia/blob/main/utils/batch_viewer.py) provided by EleutherAI
#### Use the following code to convert them into texts:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
texts = tokenizer.batch_decode(torch.load('FILE_NAME.pt'))
```
