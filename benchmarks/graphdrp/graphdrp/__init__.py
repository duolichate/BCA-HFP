"""
GraphDRP adaptation module for fair comparison with BCA-HFP.
"""

from .models.gcn import GCNNet
from .models.gat import GATNet
from .models.gat_gcn import GAT_GCN
from .models.ginconv import GINConvNet

__all__ = ['GCNNet', 'GATNet', 'GAT_GCN', 'GINConvNet']