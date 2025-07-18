import torch
import T5EmbedNetworkWeights
import pLSEmbedNetworkWeights

from ml import EmbeddingNetT5, EmbeddingNetpLS
embed_t5 = EmbeddingNetT5()
embed_pls = EmbeddingNetpLS()

# T5
#        self.fc1 = nn.Linear(input_dim, 32); self.relu1 = nn.ReLU()
#        self.fc2 = nn.Linear(32, 32);         self.relu2 = nn.ReLU()
#        self.fc3 = nn.Linear(32, emb_dim)

# PLS
#        self.fc1 = nn.Linear(input_dim, 32); self.relu1 = nn.ReLU()
#        self.fc2 = nn.Linear(32, 32);         self.relu2 = nn.ReLU()
#        self.fc3 = nn.Linear(32, emb_dim)

with torch.no_grad():
    # print(torch.tensor(T5EmbedNetworkWeights.wgtT_fc1))
    # print(torch.tensor(T5EmbedNetworkWeights.wgtT_fc1).T.shape)
    # print(embed_t5.fc1.weight.shape)
    embed_t5.fc1.weight.copy_(torch.tensor(T5EmbedNetworkWeights.wgtT_fc1).T)
    embed_t5.fc1.bias.copy_(torch.tensor(T5EmbedNetworkWeights.bias_fc1))
    embed_t5.fc2.weight.copy_(torch.tensor(T5EmbedNetworkWeights.wgtT_fc2).T)
    embed_t5.fc2.bias.copy_(torch.tensor(T5EmbedNetworkWeights.bias_fc2))
    embed_t5.fc3.weight.copy_(torch.tensor(T5EmbedNetworkWeights.wgtT_fc3).T)
    embed_t5.fc3.bias.copy_(torch.tensor(T5EmbedNetworkWeights.bias_fc3))

    embed_pls.fc1.weight.copy_(torch.tensor(pLSEmbedNetworkWeights.wgtT_fc1).T)
    embed_pls.fc1.bias.copy_(torch.tensor(pLSEmbedNetworkWeights.bias_fc1))
    embed_pls.fc2.weight.copy_(torch.tensor(pLSEmbedNetworkWeights.wgtT_fc2).T)
    embed_pls.fc2.bias.copy_(torch.tensor(pLSEmbedNetworkWeights.bias_fc2))
    embed_pls.fc3.weight.copy_(torch.tensor(pLSEmbedNetworkWeights.wgtT_fc3).T)
    embed_pls.fc3.bias.copy_(torch.tensor(pLSEmbedNetworkWeights.bias_fc3))

print(embed_t5)
print(embed_pls)

