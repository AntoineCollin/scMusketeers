import logging
logger = logging.getLogger("Sc-Musketeers")

def freeze_layers(layers_to_freeze):
    """
    Freezes specified layers in the model.

    ae: Model to freeze layers in.
    layers_to_freeze: List of layers to freeze.
    """
    for layer in layers_to_freeze:
        # logger.debug(f"Freezing layer: {layer}")
        layer.trainable = False
        if hasattr(layer, 'layers'): # If it's a nested model
            for sub_l in layer.layers:  
                sub_l.trainable = False


def freeze_block(ae, strategy):
    if strategy == "all_but_classifier_branch":
        layers_to_freeze = [
            ae.dann_discriminator,
            ae.enc,
            ae.dec,
            ae.ae_output_layer,
        ]
    elif strategy == "all_but_classifier":
        layers_to_freeze = [
            ae.dann_discriminator,
            ae.dec,
            ae.ae_output_layer,
        ]
    elif strategy == "warmup_dann":
        layers_to_freeze = []
        # layers_to_freeze = [ae.classifier]
    elif strategy == "all_but_dann_branch":
        layers_to_freeze = [
            ae.classifier,
            ae.enc,
            ae.dec,
            ae.ae_output_layer,
        ]
    elif strategy == "all_but_dann":
        layers_to_freeze = [ae.classifier, ae.dec, ae.ae_output_layer]
    elif strategy == "all_but_autoencoder":
        layers_to_freeze = [ae.classifier, ae.dann_discriminator]
    elif strategy == "freeze_dann":
        layers_to_freeze = [ae.dann_discriminator]
    elif strategy == "freeze_dec":
        layers_to_freeze = [ae.dec]
    else:
        raise ValueError("Unknown freeze strategy: " + strategy)
    return layers_to_freeze


def freeze_all(ae):
    for l in ae.layers:
        l.trainable = False


def unfreeze_all(ae):
    for layer in ae.layers:
        # logger.debug(f"Unfreezing layer: {layer}")
        layer.trainable = True
        if hasattr(layer, 'layers'): # If it's a nested model
            for sub_l in layer.layers:  
                sub_l.trainable = True
    ae.dann_discriminator.trainable = True
    for layer in ae.dann_discriminator.layers:
        # logger.debug(f"Unfreezing dann-discri layer: {layer}")
        layer.trainable = True
        if hasattr(layer, 'layers'): # If it's a nested model
            for sub_l in layer.layers:  
                sub_l.trainable = True
    ae.classifier.trainable = True
    for layer in ae.classifier.layers:
        # logger.debug(f"Unfreezing classifier layer: {layer}")
        layer.trainable = True
        if hasattr(layer, 'layers'): # If it's a nested model
            for sub_l in layer.layers:  
                sub_l.trainable = True
