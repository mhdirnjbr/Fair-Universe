import torch
import torch.nn as nn
import torch.optim as optim

class DANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, domain_dim):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, domain_dim)
        )
        self.domain_dim = domain_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.domain_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, alpha):
        x = self.feature_extractor(x)
        label_output = self.label_classifier(x)
        domain_output = self.domain_classifier(x)
        return label_output, domain_output

    def get_loss(self, x_src, y_src, x_tgt, alpha):
        src_size = x_src.size(0)
        tgt_size = x_tgt.size(0)

        # create binary labels for domain classification
        src_labels = torch.zeros(src_size, self.domain_dim)
        tgt_labels = torch.ones(tgt_size, self.domain_dim)
        domain_labels = torch.cat([src_labels, tgt_labels], dim=0)

        label_output, domain_output = self.forward(torch.cat([x_src, x_tgt], dim=0), alpha)
        label_loss = self.loss_fn(label_output[:src_size], y_src)
        domain_loss = self.domain_loss_fn(domain_output, domain_labels)
        total_loss = label_loss + domain_loss
        return total_loss