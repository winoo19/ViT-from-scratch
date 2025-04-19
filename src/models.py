# deep learning libraries
import torch


class AttentionEncoder(torch.nn.Module):
    """
    This the AttentionModel class.

    Attr:
        d_model: number of classes.
        number_of_heads: number of heads.
    """

    def __init__(
        self,
        sequence_length: int,
        d_model: int = 512,
        number_of_heads: int = 8,
        eps: float = 1e-6,
    ) -> None:
        """
        This function is the constructor of the AttentionModel class.

        Args:
            d_model: number of classes.
            number_of_heads: number of heads.
        """

        super().__init__()
        self.d_model: int = d_model
        self.number_of_heads: int = number_of_heads
        self.eps: float = eps

        self.layer_normalization_1: torch.nn.Module = LayerNormalization(
            sequence_length + 1, self.d_model
        )
        self.multi_head_attention: torch.nn.Module = MultiHeadAttention(
            self.number_of_heads, self.d_model
        )
        self.layer_normalization_2: torch.nn.Module = LayerNormalization(
            sequence_length + 1, self.d_model
        )
        self.feed_forward: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, self.d_model),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: inputs tensor. Dimensions: [batch_size, sequence_length, d_model].
        """

        x = self.layer_normalization_1(inputs)
        x = self.multi_head_attention(x) + inputs
        x_clone = x.clone()

        x = self.layer_normalization_2(x)
        x = self.feed_forward(x) + x_clone

        return x


class LayerNormalization(torch.nn.Module):
    """
    This the LayerNormalization class.

    Attr:
        d_model: number of classes.
        gamma: gamma parameter.
        beta: beta parameter.
        eps: epsilon value.
    """

    def __init__(self, sequence_length: int, d_model: int, eps: float = 1e-6) -> None:
        """
        This function is the constructor of the LayerNormalization class.

        Args:
            sequence_length: sequence length.
            d_model: number of classes.
            eps: epsilon value.
        """

        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.gamma = torch.nn.Parameter(torch.ones(self.d_model * self.sequence_length))
        self.beta = torch.nn.Parameter(torch.zeros(self.d_model * self.sequence_length))

        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: inputs tensor. Dimensions: [batch_size, sequence_length, d_model].

        Returns:
            outputs. Dimensions: [batch_size, sequence_length, d_model].
        """

        inputs = inputs.view(-1, self.d_model * self.sequence_length)

        mean = inputs.mean(dim=-1, keepdim=True)
        std = inputs.std(dim=-1, keepdim=True)

        return (self.gamma * (inputs - mean) / (std + self.eps) + self.beta).view(
            -1, self.sequence_length, self.d_model
        )


class MultiHeadAttention(torch.nn.Module):
    """
    This the MultiHeadAttention class.

    Attr:
        number_of_heads: number of heads.
        d_model: number of classes.
    """

    def __init__(self, number_of_heads: int, d_model: int) -> None:
        """
        This function is the constructor of the MultiHeadAttention class.

        Args:
            number_of_heads: number of heads.
            d_model: number of classes.
        """

        super().__init__()
        self.number_of_heads = number_of_heads
        self.d_model = d_model
        self.wq = torch.nn.Linear(self.d_model, self.d_model)
        self.wk = torch.nn.Linear(self.d_model, self.d_model)
        self.wv = torch.nn.Linear(self.d_model, self.d_model)
        self.wout = torch.nn.Linear(self.d_model, self.d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: inputs tensor. Dimensions:
            [batch_size, sequence_length, d_model].

        Returns:
            outputs. Dimensions: [batch_size, number_of_classes].
        """
        query = self.wq(inputs).view(
            inputs.shape[0],
            inputs.shape[1],
            self.number_of_heads,
            -1,
        )

        key = self.wk(inputs).view(
            inputs.shape[0],
            inputs.shape[1],
            self.number_of_heads,
            -1,
        )

        value = self.wv(inputs).view(
            inputs.shape[0],
            inputs.shape[1],
            self.number_of_heads,
            -1,
        )

        attention_weights = torch.nn.functional.softmax(
            torch.matmul(query.transpose(1, 2), key.permute(0, 2, 3, 1))
            / (self.d_model**0.5),
            dim=-1,
        )

        output = torch.matmul(attention_weights, value.transpose(1, 2)).view(
            inputs.shape[0],
            inputs.shape[1],
            -1,
        )

        return self.wout(output)


class VisionTransformer(torch.nn.Module):
    """
    This the VisionTransformer class.

    Attr:
        patch_size: patch size.
        d_model: number of classes.
        number_of_heads: number of heads.
        number_of_classes: number of classes.
        eps: epsilon value.
    """

    def __init__(
        self,
        sequence_length: int,
        patch_size: int = 16 * 16 * 3,
        d_model: int = 512,
        number_of_heads: int = 8,
        number_of_classes: int = 10,
        eps: float = 1e-6,
    ) -> None:
        """
        This function is the constructor of the VisionTransformer class.

        Args:
            patch_size: patch size.
            d_model: number of classes.
            number_of_heads: number of heads.
            number_of_classes: number of classes.
            eps: epsilon value.
        """

        super().__init__()
        self.sequence_length: int = sequence_length
        self.patch_size: int = patch_size
        self.d_model: int = d_model
        self.number_of_heads: int = number_of_heads
        self.eps: float = eps

        self.class_token: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(1, 1, self.d_model)
        )

        self.patch_embedding: torch.nn.Linear = torch.nn.Linear(
            self.patch_size, self.d_model
        )
        self.positional_embedding: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(1, self.sequence_length + 1, self.d_model)
        )
        self.attention_encoder: torch.nn.Module = AttentionEncoder(
            self.sequence_length, self.d_model, self.number_of_heads
        )

        self.wout: torch.nn.Linear = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(self.d_model, number_of_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: inputs tensor. Dimensions: [batch_size, n_patches, patch_size].

        Returns:
            outputs. Dimensions: [batch_size, number_of_classes].
        """

        x = self.patch_embedding(inputs)

        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)

        x = x + self.positional_embedding

        x = self.attention_encoder(x)

        x = x[:, 0, :].squeeze()

        x = self.wout(x)

        x = torch.nn.functional.log_softmax(x, dim=-1)

        return x
