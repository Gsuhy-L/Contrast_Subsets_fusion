from torch import nn
class AttentionalAggregation(nn.Module):
    '''
    agg attention for InterHAt
    '''
    def __init__(self, embedding_dim, hidden_dim=None):
        super(AttentionalAggregation, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim
        self.agg = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1, bias=False),
                                 nn.Softmax(dim=1))

    def forward(self, X):
        # X: b x f x emb
        attentions = self.agg(X) # b x f x 1
        attention_out = (attentions * X).sum(dim=1) # b x emb
        return attention_out
