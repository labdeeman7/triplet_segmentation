from .multitask_resnet import MultiTaskResNet
from .multitask_resnet_fpn import MultiTaskResNetFPN
from .multitask_resnet_fpn_learnable_embeddings import MultiTaskResNetFPNLearnableEmbeddings
from .multitask_resnet_fpn_masked_embeddings import MultiTaskResNetFPNMaskedEmbeddings
from .multitask_resnet_fpn_masked_embeddings_shared_transformer_decoder import MultiTaskResNetFPNMaskedEmbeddingsSharedTransformerDecoder
from .singletask_resnet_fpn import SingleTaskResNetFPN
# from .another_model import AnotherModel

__all__ = ['MultiTaskResNet', 'MultiTaskResNetFPN', 'MultiTaskResNetFPNLearnableEmbeddings',
           'MultiTaskResNetFPNMaskedEmbeddings', 'MultiTaskResNetFPNMaskedEmbeddingsSharedTransformerDecoder',
           'SingleTaskResNetFPN']
