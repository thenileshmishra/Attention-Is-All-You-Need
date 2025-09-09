BASE = {
    "N" : 6, 
    "d_model":512, 
    "d_ff":2048, 
    "h": 8, 
    "droupout": 0.1, 
    "warmup_steps": 4000, 
    "label_smoothing":0.1
}

"""
Below is the all detailed of Parameters and There role

N   Refers to the number of depth (a.k.a layers) in both encoder and decoder. 
    So:
        Encoder has 6 layers stacked.
        Decoder also has 6 layers stacked.

d_model
    The dimension of embedding and the hidden states throughout the model.
    Every token(word, subword) is represented as a 512-dimensional vector.

d_ff 
    The hidden size of the position-wise feed-forward networks(FFN) inside each layer.
    Each Transformer block has:
        1.Multi-head self-attention.
        2.Feed-forward network -> which has two linear layers:
            First Expands dimension: d_model -> d_ff (512 -> 2048)
            The projects back: d_ff -> d_model(2048 -> 512)
    So, this Created a wider "bottleneck" that lets the model capture more complex transformations.
    
h
    Number of attention heads in multi-head attention.
    Each head projects the 512-dimensional vectors into smaller subspaces (512/8 = 64 per head)
    The idea: multiple heads allow the model to look at the different parts of a sequence from different "representation subspace".

dropout
    Probability of dropping units during training.
    Prevents overfitting by adding noise and making the model more robus.

warmup_steps
    Refers to the learning rate schedule in the paper.
    Meaning:
        LR increases linearly for the first 4000 steps.
        After that, LR decays proportionally.
    This helps to stabilize early training.

label_smoothing = 0.1
    A regularization trcik for the loss function(cross-entropy).
    Instead of giving the correct class probability =1.0, they assign it = 0.9 and spread the remainging 0.1 across incorrect classes.
    Prevents the model from becoming overconfident in its predictions and improves generalization.
"""