import torch
import torch.nn.functional as F


def contrastive_loss_fn(
    image_embeddings,
    text_embeddings,
    temperature=0.07,
    return_details=False,
):
    """
    Symmetric CLIP-style contrastive loss.

    Args:
        image_embeddings:
            [B, D]

        text_embeddings:
            [B, D]

        temperature:
            Temperature used to scale similarities.

        return_details:
            If True, return individual losses and
            similarity matrix.

    Returns:
        Total contrastive loss, or a dictionary of details.
    """

    # --------------------------------------------------------
    # Normalize both modalities
    # --------------------------------------------------------

    image_embeddings = F.normalize(
        image_embeddings,
        dim=-1,
    )

    text_embeddings = F.normalize(
        text_embeddings,
        dim=-1,
    )

    # --------------------------------------------------------
    # Image-to-text similarity
    #
    # [B, D] @ [D, B]
    # -> [B, B]
    # --------------------------------------------------------

    similarity = (
        image_embeddings
        @ text_embeddings.T
    )

    # --------------------------------------------------------
    # Temperature scaling
    # --------------------------------------------------------

    logits = similarity / temperature

    # --------------------------------------------------------
    # Correct pair for sample i is caption i
    # --------------------------------------------------------

    batch_size = image_embeddings.shape[0]

    labels = torch.arange(
        batch_size,
        device=image_embeddings.device,
    )

    # --------------------------------------------------------
    # Image -> Text
    # --------------------------------------------------------

    loss_i2t = F.cross_entropy(
        logits,
        labels,
    )

    # --------------------------------------------------------
    # Text -> Image
    # --------------------------------------------------------

    loss_t2i = F.cross_entropy(
        logits.T,
        labels,
    )

    # --------------------------------------------------------
    # Symmetric CLIP loss
    # --------------------------------------------------------

    loss = (
        loss_i2t + loss_t2i
    ) / 2

    if return_details:

        return {
            "loss": loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "similarity": similarity,
            "logits": logits,
            "temperature": temperature,
        }

    return loss