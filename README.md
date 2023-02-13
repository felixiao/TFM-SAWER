# SAWER(Sequence AWare Explainable Recommendation)

## Paper

## Usage
Below are examples of how to run PETER (with and without the key feature).
```
torchrun --standalone --nproc_per_node=1 \
    SEQUER/main.py --cuda \
    --model_name sawer \
    --dataset amazon-movies
```

## Code dependencies
- Python 3.8
- PyTorch 1.6

## Code references
- [PETER](https://github.com/lileipisces/PETER)
- [Word Language Model](https://github.com/pytorch/examples/blob/master/word_language_model)
- [Sequence-to-Sequence Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Deploying a Seq2Seq Model with TorchScript](https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html)

