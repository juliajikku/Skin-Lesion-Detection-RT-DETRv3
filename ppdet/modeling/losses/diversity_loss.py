import paddle
import paddle.nn as nn


class DiversityLoss(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings):
        """
        Args:
            embeddings (Tensor):
                Shape: (num_positive_queries, hidden_dim)

        Returns:
            Tensor:
                Scalar diversity loss.
        """
        num_positive = embeddings.shape[0]

        if num_positive < 2:
            return paddle.to_tensor(0.0, dtype=embeddings.dtype)

        embeddings = nn.functional.normalize(
                     embeddings,
                    axis=1
                           )
        
        gram = paddle.matmul(
         embeddings,
         embeddings,
         transpose_y=True
                   )
        

        identity = paddle.eye(
             num_positive,
             dtype=embeddings.dtype
                 )
        
        diff = (gram - identity) ** 2

        loss = paddle.sum(diff) / (num_positive * (num_positive - 1))

        return loss